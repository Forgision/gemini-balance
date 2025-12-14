import asyncio
import os
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

from dotenv import load_dotenv
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load test environment variables
load_dotenv(dotenv_path=PROJECT_ROOT / ".env.test", override=True)

# Import settings after loading env vars
from app.config.config import settings  # noqa: E402

# Force sqlite for testing
settings.DATABASE_TYPE = "sqlite"
settings.SQLITE_DATABASE = "test.db"

from app.database.connection import Base  # noqa: E402


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_test_db():
    """
    Setup test database:
    1. Delete existing test.db if exists
    2. Create new database and tables
    3. Yield engine
    4. Cleanup
    """
    # 1. Delete existing database file
    db_path = PROJECT_ROOT / "data" / settings.SQLITE_DATABASE
    if db_path.exists():
        os.remove(db_path)

    # Ensure data directory exists
    (PROJECT_ROOT / "data").mkdir(exist_ok=True)

    # 2. Create async engine for testing
    # Use StaticPool for in-memory/file sqlite to avoid threading issues in tests if needed,
    # but for file-based sqlite, standard pool is usually fine.
    # However, since we want to share the connection or ensure isolation, let's stick to standard but ensure clean state.

    # Re-construct database URL to be sure it points to the right place
    db_url = f"sqlite+aiosqlite:///{db_path}"

    engine = create_async_engine(
        db_url, connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()
    # Optional: remove db file after tests
    # if db_path.exists():
    #     os.remove(db_path)


@pytest.fixture(autouse=True)
async def db_session(setup_test_db):
    """
    Fixture to provide a fresh database session for each test.
    Also patches app.database.connection.AsyncSessionLocal to use the test engine.
    """
    engine = setup_test_db

    # Create a new session factory bound to the test engine
    TestingSessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )

    # Create a session
    session = TestingSessionLocal()

    # Patch the AsyncSessionLocal in connection module to return our session factory
    # But wait, AsyncSessionLocal is a class/object. We should patch it or the engine it uses.
    # Easier: Patch get_db dependency or Patch AsyncSessionLocal to return this session.

    # Let's try patching AsyncSessionLocal to just return this session when called?
    # No, AsyncSessionLocal() creates a session.

    # Better approach: Override the get_db dependency in FastAPI app (if using FastAPI TestClient)
    # AND Patch AsyncSessionLocal for non-FastAPI calls (service layer tests).

    # Patch AsyncSessionLocal in app.database.connection to return our testing session factory
    with patch("app.database.connection.AsyncSessionLocal", TestingSessionLocal):
        yield session
        await session.rollback()  # Rollback changes after each test to keep DB clean
        await session.close()


from fastapi.testclient import TestClient  # noqa: E402
from app.core.application import create_app  # noqa: E402
from app.dependencies import get_key_manager  # noqa: E402
from tests.fixtures.mock_api import mock_external_apis  # noqa: E402, F401

TEST_AUTH_TOKEN = "test_auth_token_12345"


@pytest.fixture(scope="session")
def auth_token():
    return TEST_AUTH_TOKEN


@pytest.fixture(scope="session")
def goog_api_key_header(auth_token):
    return {"x-goog-api-key": auth_token}


@pytest.fixture(scope="session", autouse=True)
def mock_env_import():
    """
    Prevent importing env vars to DB during tests.
    """
    with patch("app.core.application.sync_initial_settings", new=AsyncMock()) as mock:
        yield mock


@pytest.fixture(scope="session")
def test_app(setup_test_db, mock_external_apis, mock_env_import):  # noqa: F811
    """
    Create a test app instance.
    """

    # Mock KeyManager dependency
    async def mock_get_key_manager():
        mock_km = AsyncMock()
        mock_km.is_ready = True
        mock_km.get_key.return_value = "mock_api_key"
        mock_km.get_random_valid_key.return_value = "mock_api_key"
        mock_km.handle_api_failure.return_value = "mock_api_key"
        mock_km.update_usage.return_value = True
        mock_km.reset_key_failure_count.return_value = True
        mock_km.get_state.return_value = {"models": {}, "summary": {}}
        mock_km.get_keys_by_status.return_value = {"valid_keys": {}, "invalid_keys": {}}
        return mock_km

    # Set auth token in settings for verification
    settings.AUTH_TOKEN = TEST_AUTH_TOKEN
    settings.ALLOWED_TOKENS = [TEST_AUTH_TOKEN]
    print(f"DEBUG CONFTEST: id(settings)={id(settings)}")
    print(f"DEBUG CONFTEST: settings.AUTH_TOKEN set to '{settings.AUTH_TOKEN}'")

    app = create_app()
    app.dependency_overrides[get_key_manager] = mock_get_key_manager

    print(
        f"DEBUG CONFTEST: after create_app, settings.AUTH_TOKEN='{settings.AUTH_TOKEN}'"
    )
    return app


@pytest.fixture(scope="session")
def test_client(test_app):
    """
    Create a test client.
    """
    with TestClient(test_app) as client:
        yield client


def mock_check_for_updates():
    """
    Mock the check_for_updates function to prevent network calls to GitHub.
    """
    with patch(
        "app.service.update.update_service.check_for_updates", new_callable=AsyncMock
    ) as mock:
        mock.return_value = (False, None, None)
        yield mock
