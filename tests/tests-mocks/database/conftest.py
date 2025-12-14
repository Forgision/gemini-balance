import pytest
import pytest_asyncio
from pytest import MonkeyPatch
import importlib

from app.config.config import settings
from sqlalchemy.orm import sessionmaker, Session


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch fixture for database tests."""
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest_asyncio.fixture(scope="session")
async def db_engine(monkeypatch_session):
    """
    Session-scoped fixture to set up and tear down an in-memory SQLite database.
    Returns a SQLAlchemy engine.
    Used specifically for database tests.
    """
    monkeypatch_session.setattr(settings, "SQLITE_DATABASE", ":memory:")
    monkeypatch_session.setattr(settings, "DATABASE_TYPE", "sqlite")

    # Reload the connection module to apply the new settings
    from app.database import connection

    importlib.reload(connection)

    # Now we can import the engine and Base
    from app.database.connection import Base, engine

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(scope="function")
def db_session(db_engine):
    """
    Function-scoped fixture to provide a transactional session for each test.
    Rolls back the transaction after the test is complete.
    Used specifically for database tests.
    """
    SessionLocal = sessionmaker(db_engine, class_=Session, expire_on_commit=False)

    with SessionLocal() as session:
        yield session
        session.rollback()
