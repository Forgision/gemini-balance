"""
Database initialization module
"""

from dotenv import dotenv_values

from sqlalchemy import inspect as sqlalchemy_inspect

from app.database.connection import engine, Base, AsyncSessionLocal
from app.database.models import Settings
from app.log.logger import get_database_logger

logger = get_database_logger()


async def create_tables():
    """
    Create database tables using async engine
    """
    try:
        # Create all tables using async engine
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise


async def import_env_to_settings():
    """
    Import configuration items from the .env file into the t_settings table
    """
    try:
        # Get all configuration items from the .env file
        env_values = dotenv_values(".env")

        # Check if the t_settings table exists using async engine
        async with engine.begin() as conn:
            # Use SQLAlchemy 2.0 run_sync() for inspection in async context
            table_names = await conn.run_sync(
                lambda sync_conn: sqlalchemy_inspect(sync_conn).get_table_names()
            )

        # Check if the t_settings table exists
        if "t_settings" in table_names:
            # Use an async session for database operations
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select

                # Get all existing configuration items
                result = await session.execute(select(Settings))
                current_settings = {
                    setting.key: setting for setting in result.scalars().all()
                }

                # Iterate over all configuration items
                for key, value in env_values.items():
                    # Check if the configuration item already exists
                    if key not in current_settings:
                        # Insert the configuration item
                        new_setting = Settings(key=key, value=value)
                        session.add(new_setting)
                        logger.info(f"Inserted setting: {key}")

                # Commit the transaction
                await session.commit()
        else:
            logger.error(
                "t_settings table does not exist, skipping import of environment variables"
            )

        logger.info("Environment variables imported to settings table successfully")
    except Exception as e:
        logger.error(
            f"Failed to import environment variables to settings table: {str(e)}"
        )
        raise


async def initialize_database():
    """
    Initialize the database
    """
    try:
        # Create tables
        await create_tables()

        # Import environment variables
        await import_env_to_settings()
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
