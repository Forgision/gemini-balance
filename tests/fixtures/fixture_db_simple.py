import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from unittest.mock import patch

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
    db_url = f"sqlite+aiosqlite:///{db_path}"

    engine = create_async_engine(
        db_url, connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


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

    # Patch AsyncSessionLocal in app.database.connection to return our testing session factory
    with patch("app.database.connection.AsyncSessionLocal", TestingSessionLocal):
        yield session
        await session.rollback()  # Rollback changes after each test to keep DB clean
        await session.close()
