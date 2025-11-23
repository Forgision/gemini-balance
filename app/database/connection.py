"""
Database connection pool module
"""

from pathlib import Path
from typing import AsyncGenerator
import platform
from urllib.parse import quote_plus
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.config.config import settings
from app.log.logger import get_database_logger

logger = get_database_logger()

# Database URL - using async drivers
if settings.DATABASE_TYPE == "sqlite":
    # Handle in-memory database specially
    if settings.SQLITE_DATABASE == ":memory:":
        DATABASE_URL = "sqlite+aiosqlite:///:memory:"
    else:
        # Ensure the data directory exists
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        db_path = data_dir / settings.SQLITE_DATABASE
        # Following is to avoid windows separator in windows
        db_path_str = db_path.as_posix()
        DATABASE_URL = f"sqlite+aiosqlite:///{db_path_str}"
elif settings.DATABASE_TYPE == "mysql":
    is_windows = platform.system() == "Windows"
    if settings.MYSQL_SOCKET and not is_windows:
        DATABASE_URL = f"mysql+aiomysql://{settings.MYSQL_USER}:{quote_plus(settings.MYSQL_PASSWORD)}@/{settings.MYSQL_DATABASE}?unix_socket={settings.MYSQL_SOCKET}"
    else:
        DATABASE_URL = f"mysql+aiomysql://{settings.MYSQL_USER}:{quote_plus(settings.MYSQL_PASSWORD)}@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
else:
    raise ValueError(
        "Unsupported database type. Please set DATABASE_TYPE to 'sqlite' or 'mysql'."
    )

# Create async database engine
# pool_pre_ping=True: Perform a simple "ping" test before getting a connection from the pool to ensure the connection is valid
# For MySQL, configure pool settings
pool_kwargs: dict = {"pool_pre_ping": True}
if settings.DATABASE_TYPE == "mysql":
    pool_kwargs.update({
        "pool_size": 5,
        "max_overflow": 20,
        "pool_recycle": 1800,  # Recycle connections after 30 minutes
    })

engine: AsyncEngine = create_async_engine(DATABASE_URL, **pool_kwargs)

# Create base class using SQLAlchemy 2.0 style
class Base(DeclarativeBase):
    """Base class for all database models"""
    pass

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI route handlers to get database session.
    
    Yields:
        AsyncSession: Database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def connect_to_db():
    """
    Connect to the database (verify engine is ready).
    For async engines, this just verifies connectivity.
    """
    try:
        # Test connection by creating a session and executing a simple query
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        logger.info(f"Connected to {settings.DATABASE_TYPE}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}", exc_info=True)
        raise


async def disconnect_from_db():
    """
    Disconnect from the database (dispose engine).
    """
    try:
        await engine.dispose()
        logger.info(f"Disconnected from {settings.DATABASE_TYPE}")
    except Exception as e:
        logger.error(f"Failed to disconnect from database: {str(e)}", exc_info=True)
