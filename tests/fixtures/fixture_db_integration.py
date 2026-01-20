import pytest
import asyncio
from pathlib import Path
from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from app.config.config import settings

from tests.fixtures.fixture_consts import (
    TEST_API_KEYS,
    TEST_VERTEX_API_KEYS,
    TEST_AUTH_TOKEN,
    TEST_ALLOWED_TOKENS,
)


def run_async_safe(coro):
    """
    Run an async coroutine safely, handling both cases:
    - When there's a running event loop (e.g., in async tests)
    - When there's no running event loop (e.g., in session fixtures)
    """
    try:
        # Try to get the running event loop
        asyncio.get_running_loop()
        # If we're here, there's a running loop
        # We need to run this in a new thread with a new event loop
        import concurrent.futures

        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        # No running event loop, safe to use asyncio.run()
        return asyncio.run(coro)


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch fixture for database settings."""
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def patch_database_settings(monkeypatch_session):
    """Patch database settings to use in-memory SQLite for both main DB and KeyManager DB."""
    # Patch main database
    monkeypatch_session.setattr(settings, "DATABASE_TYPE", "sqlite")
    monkeypatch_session.setattr(settings, "SQLITE_DATABASE", "integration_test.sqlite")

    # Patch test configuration
    monkeypatch_session.setattr(settings, "API_KEYS", TEST_API_KEYS)
    monkeypatch_session.setattr(settings, "VERTEX_API_KEYS", TEST_VERTEX_API_KEYS)
    monkeypatch_session.setattr(settings, "AUTH_TOKEN", TEST_AUTH_TOKEN)
    monkeypatch_session.setattr(settings, "ALLOWED_TOKENS", TEST_ALLOWED_TOKENS)


@pytest.fixture(scope="session")
def in_memory_db_engine(patch_database_settings, monkeypatch_session):
    """Session-scoped fixture that patches the app to use a shared async in-memory SQLite engine."""
    from app.database.connection import Base
    import app.database.connection as db_conn

    db_dir = Path("data")
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "integration_test.sqlite"
    if db_path.exists():
        db_path.unlink()

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    run_async_safe(create_tables())

    session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    # Patch the application-level engine/session so all code paths share the same DB
    monkeypatch_session.setattr(db_conn, "engine", engine, raising=False)
    monkeypatch_session.setattr(
        db_conn, "AsyncSessionLocal", session_maker, raising=False
    )

    # Dynamic patching: Iterate over all loaded modules and patch AsyncSessionLocal and engine
    # This ensures that any module that imported them at top level gets the patched version
    import sys

    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("app."):
            if hasattr(module, "AsyncSessionLocal"):
                monkeypatch_session.setattr(
                    module, "AsyncSessionLocal", session_maker, raising=False
                )
            if hasattr(module, "engine"):
                monkeypatch_session.setattr(module, "engine", engine, raising=False)

    yield engine

    async def dispose_engine():
        await engine.dispose()

    run_async_safe(dispose_engine())
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(scope="session")
def key_manager_async_engine(patch_database_settings):
    """Session-scoped fixture to create in-memory async SQLite engine for KeyManager."""
    from app.database.connection import Base

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    # Create tables
    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    run_async_safe(create_tables())

    yield engine

    # Cleanup
    async def dispose_engine():
        await engine.dispose(close=True)

    run_async_safe(dispose_engine())
