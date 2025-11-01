import pytest
from pytest import MonkeyPatch
import importlib

from app.config.config import settings
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="session")
def db_engine():
    """
    Session-scoped fixture to set up and tear down an in-memory SQLite database.
    Returns a SQLAlchemy engine.
    """
    mp = MonkeyPatch()
    mp.setattr(settings, "SQLITE_DATABASE", ":memory:")
    mp.setattr(settings, "DATABASE_TYPE", "sqlite")

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
    mp.undo()


@pytest.fixture(scope="session")
def db_session(db_engine):
    """
    Function-scoped fixture to provide a transactional session for each test.
    Rolls back the transaction after the test is complete.
    """
    connection = db_engine.connect()

    # Begin a transaction
    trans = connection.begin()

    # Bind an individual session to the connection
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    # Rollback the transaction and close the connection
    session.close()
    trans.rollback()
    connection.close()
