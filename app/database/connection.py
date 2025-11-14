"""
Database connection pool module
"""

from pathlib import Path, PureWindowsPath
from urllib.parse import quote_plus
from databases import Database
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base

from app.config.config import settings
from app.log.logger import get_database_logger

logger = get_database_logger()

# Database URL
if settings.DATABASE_TYPE == "sqlite":
    # Handle in-memory database specially
    if settings.SQLITE_DATABASE == ":memory:":
        DATABASE_URL = "sqlite:///:memory:"
    else:
        # Ensure the data directory exists
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        db_path = data_dir / settings.SQLITE_DATABASE
        # Following is to avoid windows separator in windows
        db_path = PureWindowsPath(db_path).as_posix()
        DATABASE_URL = f"sqlite:///{db_path}"
elif settings.DATABASE_TYPE == "mysql":
    if settings.MYSQL_SOCKET:
        DATABASE_URL = f"mysql+pymysql://{settings.MYSQL_USER}:{quote_plus(settings.MYSQL_PASSWORD)}@/{settings.MYSQL_DATABASE}?unix_socket={settings.MYSQL_SOCKET}"
    else:
        DATABASE_URL = f"mysql+pymysql://{settings.MYSQL_USER}:{quote_plus(settings.MYSQL_PASSWORD)}@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
else:
    raise ValueError(
        "Unsupported database type. Please set DATABASE_TYPE to 'sqlite' or 'mysql'."
    )

# Create database engine
# pool_pre_ping=True: Perform a simple "ping" test before getting a connection from the pool to ensure the connection is valid
# TODO: implement sqlite+aiosqlite
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create metadata object
metadata = MetaData()

# Create base class
Base = declarative_base(metadata=metadata)

# Create a database connection pool and configure its parameters; connection pooling is not used in SQLite
# min_size/max_size: The minimum/maximum number of connections in the connection pool
# pool_recycle=3600: The maximum number of seconds a connection is allowed to exist in the pool (lifecycle).
#                    Set to 3600 seconds (1 hour) to ensure connections are recycled before the default MySQL wait_timeout (usually 8 hours) or other network timeouts.
#                    If you encounter connection failure issues, you can try lowering this value to be less than the actual wait_timeout or network timeout.
# The databases library automatically handles reconnection attempts after a connection failure.
if settings.DATABASE_TYPE == "sqlite":
    database = Database(DATABASE_URL)
else:
    database = Database(DATABASE_URL, min_size=5, max_size=20, pool_recycle=1800)


async def connect_to_db():
    """
    Connect to the database
    """
    try:
        if not database.is_connected:
            await database.connect()
            logger.info(f"Connected to {settings.DATABASE_TYPE}")
        else:
            logger.debug(f"Already connected to {settings.DATABASE_TYPE}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise


async def disconnect_from_db():
    """
    Disconnect from the database
    """
    try:
        await database.disconnect()
        logger.info(f"Disconnected from {settings.DATABASE_TYPE}")
    except Exception as e:
        logger.error(f"Failed to disconnect from database: {str(e)}")
