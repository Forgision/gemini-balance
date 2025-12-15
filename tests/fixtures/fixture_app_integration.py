import asyncio
import json
import datetime
import pytest
import pytest_asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, AsyncMock, patch

from sqlalchemy import select, insert, update, inspect as sqlalchemy_inspect
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config.config import settings
from app.service.key.key_manager import KeyManager
from app.dependencies import get_key_manager
from tests.fixtures.fixture_consts import (
    TEST_API_KEYS,
    TEST_VERTEX_API_KEYS,
    TEST_AUTH_TOKEN,
    TEST_ALLOWED_TOKENS,
)


@pytest_asyncio.fixture(scope="function")
async def test_key_manager(key_manager_async_engine):
    """Function-scoped fixture to create and initialize a real KeyManager instance."""
    # Create new async session maker for this test
    async_session_maker = async_sessionmaker(
        key_manager_async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    # Create tables for KeyManager database before initializing
    from app.database.connection import Base

    async def create_key_manager_tables():
        """Create KeyManager tables using the test engine."""
        async with key_manager_async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Create tables before KeyManager initialization
    await create_key_manager_tables()

    # Patch asyncio.create_task to ensure tasks are created in the current event loop
    original_create_task = asyncio.create_task
    current_loop = asyncio.get_event_loop()

    def patched_create_task(coro, name=None):
        """Create task in the current event loop."""
        try:
            # Ensure we're using the current loop
            loop = asyncio.get_running_loop()
            return loop.create_task(coro, name=name)
        except RuntimeError:
            # If no running loop, use the current loop
            return current_loop.create_task(coro, name=name)

    # Temporarily patch asyncio.create_task
    import asyncio as asyncio_module

    asyncio_module.create_task = patched_create_task

    try:
        # Create KeyManager instance
        key_manager = KeyManager(
            api_keys=TEST_API_KEYS,
            vertex_api_keys=TEST_VERTEX_API_KEYS,
            async_session_maker=async_session_maker,
        )

        # Initialize KeyManager
        default_rate_limits = {
            "gemini-pro": {"RPM": 60, "TPM": 1000000, "RPD": 1500},
            "gemini-2.0-flash-exp": {"RPM": 15, "TPM": 1000000, "RPD": 1500},
            "gemini-2.5-pro": {"RPM": 60, "TPM": 1000000, "RPD": 1500},
            "gemini-2.5-flash": {"RPM": 60, "TPM": 1000000, "RPD": 1500},
        }

        try:
            await key_manager.init(rate_limit_data=default_rate_limits)
        except Exception:
            # If init fails, try with default
            await key_manager.init()

        # Wait a small amount to ensure initialization is complete
        await asyncio.sleep(0.1)

        yield key_manager

        # Cleanup - ensure background task is properly stopped
        try:
            if key_manager._background_task and not key_manager._background_task.done():
                key_manager._stop_event.set()
                key_manager._background_task.cancel()
                try:
                    await asyncio.wait_for(key_manager._background_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            await key_manager.shutdown()
        except Exception:
            pass
    finally:
        # Restore original functions
        asyncio_module.create_task = original_create_task


@pytest_asyncio.fixture(scope="function")
async def test_app(
    in_memory_db_engine, test_key_manager, patched_api_clients, patched_service_clients
):
    """Function-scoped fixture to create a test FastAPI app using the real create_app() function."""
    # Ensure KeyManager is initialized (it should be from test_key_manager fixture, but verify)
    if not test_key_manager.is_ready:
        await test_key_manager.init()

    # FULLY initialize database with required config BEFORE create_app() runs
    from app.database.connection import AsyncSessionLocal, connect_to_db
    from app.database.initialization import initialize_database
    from app.config.config import sync_initial_settings
    from app.database.models import Settings as SettingsModel

    # Clear all tables to ensure clean state for each test
    from app.database.connection import engine
    from sqlalchemy import text

    async with engine.begin() as conn:
        # Disable foreign key checks to allow truncation
        await conn.execute(text("PRAGMA foreign_keys = OFF"))

        # Get all table names
        tables = await conn.run_sync(
            lambda sync_conn: sqlalchemy_inspect(sync_conn).get_table_names()
        )

        for table in tables:
            # Use DELETE FROM instead of TRUNCATE for SQLite
            await conn.execute(text(f"DELETE FROM {table}"))

        # Re-enable foreign keys
        await conn.execute(text("PRAGMA foreign_keys = ON"))

    # Ensure tables exist
    from app.database.connection import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Step 1: Initialize database
    await initialize_database()

    # Step 2: Connect to database
    await connect_to_db()

    # Step 3: Sync initial settings first
    await sync_initial_settings()

    # Step 4: Ensure TEST_AUTH_TOKEN and TEST_ALLOWED_TOKENS are in database AFTER sync
    try:
        async with AsyncSessionLocal() as session:
            # Check if AUTH_TOKEN exists in database
            query = select(SettingsModel.key, SettingsModel.value).where(
                SettingsModel.key.in_(["AUTH_TOKEN", "ALLOWED_TOKENS"])
            )
            result = await session.execute(query)
            rows = result.fetchall()
            existing = {
                dict(row._mapping)["key"]: dict(row._mapping)["value"] for row in rows
            }

            now = datetime.datetime.now(datetime.timezone.utc)

            # Insert or update AUTH_TOKEN
            if "AUTH_TOKEN" not in existing:
                query_insert = insert(SettingsModel).values(
                    key="AUTH_TOKEN",
                    value=TEST_AUTH_TOKEN,
                    description="AUTH_TOKEN configuration setting",
                    created_at=now,
                    updated_at=now,
                )
                await session.execute(query_insert)
                await session.commit()
            elif existing["AUTH_TOKEN"] != TEST_AUTH_TOKEN:
                query_update = (
                    update(SettingsModel)
                    .where(SettingsModel.key == "AUTH_TOKEN")
                    .values(value=TEST_AUTH_TOKEN, updated_at=now)
                )
                await session.execute(query_update)
                await session.commit()

            # Insert or update ALLOWED_TOKENS
            allowed_tokens_json = json.dumps(TEST_ALLOWED_TOKENS, ensure_ascii=False)
            if "ALLOWED_TOKENS" not in existing:
                query_insert = insert(SettingsModel).values(
                    key="ALLOWED_TOKENS",
                    value=allowed_tokens_json,
                    description="ALLOWED_TOKENS configuration setting",
                    created_at=now,
                    updated_at=now,
                )
                await session.execute(query_insert)
                await session.commit()
            elif existing["ALLOWED_TOKENS"] != allowed_tokens_json:
                query_update = (
                    update(SettingsModel)
                    .where(SettingsModel.key == "ALLOWED_TOKENS")
                    .values(value=allowed_tokens_json, updated_at=now)
                )
                await session.execute(query_update)
                await session.commit()

        # Re-sync to load our test tokens into memory settings
        await sync_initial_settings()

    except Exception:
        # If database setup fails, ensure settings are still patched
        pass

    # Step 5: Ensure settings object ALWAYS has test tokens (final override)
    settings.AUTH_TOKEN = TEST_AUTH_TOKEN
    settings.ALLOWED_TOKENS = TEST_ALLOWED_TOKENS

    # Create app using the real create_app() function
    from app.core.application import create_app

    # Override lifespan to skip database reinitialization
    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        """Test lifespan that skips database init (already done) and uses test KeyManager."""
        from app.database.connection import disconnect_from_db

        app.state.key_manager = test_key_manager

        yield

        await disconnect_from_db()

    # Temporarily patch create_app to use test lifespan
    from app.core import application as app_module

    original_lifespan = app_module.lifespan
    app_module.lifespan = test_lifespan

    try:
        app = create_app()
        app.state.key_manager = test_key_manager
    finally:
        # Restore original lifespan
        app_module.lifespan = original_lifespan

    # Override get_key_manager dependency
    from fastapi import Request as FastAPIRequest

    async def override_get_key_manager(request=None):
        return test_key_manager

    app.dependency_overrides[get_key_manager] = override_get_key_manager

    # Override security dependencies globally
    from fastapi import Header, HTTPException, Query
    from typing import Optional
    from app.core.security import verify_auth_token

    # Mock for API Key auth
    async def mock_security_dependency(key: str | None = Query(None), x_goog_api_key: str | None = Header(None)):
        # Always return valid token to pass auth check in tests
        return TEST_AUTH_TOKEN

    # Mock for Bearer Token auth
    async def mock_verify_authorization(authorization: Optional[str] = Header(None)):
        return TEST_AUTH_TOKEN

    # Mock for Header Token auth
    async def mock_verify_auth_token_method(
        authorization: Optional[str] = Header(None),
    ):
        return TEST_AUTH_TOKEN

    # Patch verify_auth_token function used in cookies
    def mock_verify_auth_token_func(token: str) -> bool:
        return True # Always pass for tests

    # Monkeypatch the standalone function in all modules where it is used
    import app.core.security as security_module
    original_verify_auth_token = security_module.verify_auth_token
    security_module.verify_auth_token = mock_verify_auth_token_func

    import app.middleware.middleware as middleware_module
    middleware_module.verify_auth_token = mock_verify_auth_token_func

    # Patch in routers that import it
    try:
        import app.router.error_log_routes as error_log_routes_module
        error_log_routes_module.verify_auth_token = mock_verify_auth_token_func
    except ImportError: pass

    try:
        import app.router.config_routes as config_routes_module
        config_routes_module.verify_auth_token = mock_verify_auth_token_func
    except ImportError: pass

    try:
        import app.router.key_routes as key_routes_module
        key_routes_module.verify_auth_token = mock_verify_auth_token_func
    except ImportError: pass

    try:
        import app.router.scheduler_routes as scheduler_routes_module
        scheduler_routes_module.verify_auth_token = mock_verify_auth_token_func
    except ImportError: pass

    try:
        import app.router.stats_routes as stats_routes_module
        stats_routes_module.verify_auth_token = mock_verify_auth_token_func
    except ImportError: pass

    # Apply overrides to SecurityService methods used in routes

    # gemini_routes
    app.dependency_overrides[
        gemini_routes.security_service.verify_key_or_goog_api_key
    ] = mock_security_dependency

    # openai_compatible_routes
    app.dependency_overrides[
        openai_compatible_routes.security_service.verify_authorization
    ] = mock_verify_authorization
    app.dependency_overrides[
        openai_compatible_routes.security_service.verify_auth_token
    ] = mock_verify_auth_token_method

    # openai_routes
    app.dependency_overrides[
        openai_routes.security_service.verify_authorization
    ] = mock_verify_authorization
    app.dependency_overrides[
        openai_routes.security_service.verify_auth_token
    ] = mock_verify_auth_token_method

    # vertex_express_routes
    app.dependency_overrides[
        vertex_express_routes.security_service.verify_key_or_goog_api_key
    ] = mock_security_dependency

    # claude_routes
    app.dependency_overrides[claude_routes.security_service.verify_auth_token] = (
        mock_verify_auth_token_method
    )

    try:
        from app.router import files_routes
        app.dependency_overrides[
            files_routes.security_service.verify_key_or_goog_api_key
        ] = mock_security_dependency
    except ImportError:
        pass

    # Override dependency for scheduler/stats routes dependencies
    async def mock_verify_token_dep(request: FastAPIRequest):
        pass # No exception = authorized

    if hasattr(scheduler_routes, "verify_token"):
        app.dependency_overrides[scheduler_routes.verify_token] = mock_verify_token_dep

    if hasattr(stats_routes, "verify_token"):
        app.dependency_overrides[stats_routes.verify_token] = mock_verify_token_dep

    app.state.key_manager = test_key_manager

    # Patch scheduler stop function
    from app.scheduler import scheduled_tasks as scheduler_module
    from app.log.logger import Logger

    scheduler_logger = Logger.setup_logger("scheduler")
    original_stop_scheduler = scheduler_module.stop_scheduler

    def patched_stop_scheduler():
        try:
            return original_stop_scheduler()
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                scheduler_logger.info(
                    "Scheduler event loop is closed (likely already stopped)."
                )
                scheduler_module.scheduler_instance = None
                return
            raise

    scheduler_module.stop_scheduler = patched_stop_scheduler

    try:
        yield app
    finally:
        scheduler_module.stop_scheduler = original_stop_scheduler
        scheduler_module.scheduler_instance = None
        from app.database.connection import disconnect_from_db
        await disconnect_from_db()
        app.dependency_overrides.clear()
        security_module.verify_auth_token = original_verify_auth_token


@pytest.fixture(scope="function")
def test_client(test_app):
    """Function-scoped fixture to create a TestClient for the test app."""
    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture(scope="function")
def auth_header(auth_token):
    """Function-scoped fixture providing Authorization header."""
    return {"Authorization": f"Bearer {auth_token}"}

@pytest.fixture(scope="function")
def auth_cookies(auth_token):
    """Function-scoped fixture providing auth_token cookie for cookie-based authentication."""
    return {"auth_token": auth_token}
