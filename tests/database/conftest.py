import pytest
from pytest import MonkeyPatch
import importlib

from app.config.config import settings
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch fixture for database tests."""
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def db_engine(monkeypatch_session):
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
    Base.metadata.create_all(bind=engine)

    yield engine

    # Drop tables
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
async def db_session(db_engine):
    """
    Function-scoped fixture to provide a transactional session for each test.
    Rolls back the transaction after the test is complete.
    Used specifically for database tests.
    """
    async_session = sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session
        await session.rollback()

