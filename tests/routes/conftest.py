import pytest
from pytest import MonkeyPatch
import importlib

from app.config.config import settings
from fastapi.testclient import TestClient

from app.core.application import create_app
from app.dependencies import get_error_log_service
from app.router import (
    gemini_routes,
    openai_routes,
    vertex_express_routes,
    openai_compatible_routes,
    key_routes,
    claude_routes,
)


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch fixture for route tests."""
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def route_db_engine(monkeypatch_session):
    """
    Session-scoped fixture to set up and tear down an in-memory SQLite database.
    Used specifically for route tests that need a database.
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


@pytest.fixture(scope="session")
def route_async_db_engine(monkeypatch_session):
    """
    Session-scoped fixture to set up and tear down an in-memory async SQLite database.
    Used for the UsageMatrix table in KeyManager.
    """
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool
    
    # Patch the async database URL to use in-memory database
    monkeypatch_session.setattr(settings, "KEY_MATRIX_DB_URL", "sqlite+aiosqlite:///:memory:")
    
    # Reload the key_manager module to apply the new settings
    from app.service.key import key_manager
    importlib.reload(key_manager)
    
    # Import the engine and Base after reload
    from app.service.key.key_manager import engine, Base
    
    # Create tables in the async database
    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    # Run the async table creation
    asyncio.run(create_tables())
    
    yield engine
    
    # Cleanup: dispose of the engine
    async def dispose_engine():
        await engine.dispose(close=True)
    
    asyncio.run(dispose_engine())


@pytest.fixture(scope="session")
def route_mock_key_manager():
    """
    Session-scoped mock KeyManager for route tests.
    Provides better performance by sharing the mock across route tests.
    """
    from unittest.mock import MagicMock, AsyncMock
    
    mock = MagicMock()
    mock.get_random_valid_key = AsyncMock(return_value="test_api_key")
    mock.get_next_working_key = AsyncMock(return_value="test_api_key_for_model")
    mock.get_paid_key = AsyncMock(return_value="test_paid_api_key")
    mock.get_key = AsyncMock(return_value="test_api_key_for_model")
    mock.get_all_keys_with_fail_count = AsyncMock(return_value={"valid_keys": {}, "invalid_keys": {}})
    mock.handle_api_failure = AsyncMock(return_value=None)
    return mock


@pytest.fixture(scope="session")
def route_mock_error_log_service():
    """
    Session-scoped mock error_log_service for route tests.
    Provides better performance by sharing the mock across route tests.
    """
    from unittest.mock import AsyncMock
    
    mock = AsyncMock()
    mock.process_get_error_logs.return_value = {"logs": [], "total": 0}
    mock.process_get_error_log_details.return_value = {}
    mock.process_find_error_log_by_info.return_value = {}
    mock.process_delete_error_logs_by_ids.return_value = 1
    mock.process_delete_all_error_logs.return_value = None
    mock.process_delete_error_log_by_id.return_value = True
    return mock


@pytest.fixture(scope="session")
def route_mock_chat_service():
    """
    Session-scoped mock chat service for route tests.
    Provides better performance by sharing the mock across route tests.
    """
    from unittest.mock import MagicMock, AsyncMock
    
    mock = MagicMock()
    mock.generate_content = AsyncMock(return_value={"candidates": [{"content": {"parts": [{"text": "Hello, world!"}]}}]})
    mock.count_tokens = AsyncMock(return_value={"totalTokens": 123})
    return mock


@pytest.fixture(scope="session")
def route_mock_embedding_service():
    """
    Session-scoped mock embedding service for route tests.
    Provides better performance by sharing the mock across route tests.
    """
    from unittest.mock import MagicMock, AsyncMock
    
    mock = MagicMock()
    mock.embed_content = AsyncMock(return_value={"embedding": {"values": [0.1, 0.2, 0.3]}})
    return mock


@pytest.fixture(scope="session")
def route_mock_config_service():
    """
    Session-scoped mock ConfigService for route tests.
    Provides better performance by sharing the mock across route tests.
    """
    from unittest.mock import MagicMock, AsyncMock
    
    mock = MagicMock()
    mock.get_config = AsyncMock(return_value={"LOG_LEVEL": "INFO", "API_KEYS": []})
    mock.update_config = AsyncMock(return_value={"status": "updated"})
    mock.reset_config = AsyncMock(return_value={"status": "reset"})
    mock.delete_key = AsyncMock(return_value={"success": True})
    mock.delete_selected_keys = AsyncMock(return_value={"success": True, "deleted_count": 0})
    mock.fetch_ui_models = AsyncMock(return_value={"models": []})
    return mock


@pytest.fixture(scope="session")
def route_mock_proxy_check_service():
    """
    Session-scoped mock ProxyCheckService for route tests.
    Provides better performance by sharing the mock across route tests.
    """
    from unittest.mock import MagicMock, AsyncMock
    import time
    
    mock = MagicMock()
    mock.check_single_proxy = AsyncMock(return_value={
        "proxy": "proxy1",
        "is_available": True,
        "response_time": 0.5,
        "error_message": None,
        "checked_at": time.time(),
    })
    mock.check_multiple_proxies = AsyncMock(return_value=[])
    mock.get_cache_stats = MagicMock(return_value={"hits": 0, "misses": 0})
    mock.clear_cache = MagicMock(return_value=None)
    return mock


@pytest.fixture(scope="function", autouse=True)
def reset_route_mocks(
    route_mock_key_manager,
    route_mock_error_log_service,
    route_mock_chat_service,
    route_mock_embedding_service,
    route_mock_config_service,
    route_mock_proxy_check_service,
):
    """
    Function-scoped fixture to reset route mocks between tests.
    Ensures clean state while maintaining session-scoped mocks for performance.
    """
    # Reset key_manager mock
    route_mock_key_manager.reset_mock()
    route_mock_key_manager.get_random_valid_key.return_value = "test_api_key"
    route_mock_key_manager.get_next_working_key.return_value = "test_api_key_for_model"
    route_mock_key_manager.get_paid_key.return_value = "test_paid_api_key"
    route_mock_key_manager.get_key.return_value = "test_api_key_for_model"
    route_mock_key_manager.get_all_keys_with_fail_count.return_value = {"valid_keys": {}, "invalid_keys": {}}
    route_mock_key_manager.handle_api_failure.return_value = None
    
    # Reset error_log_service mock
    route_mock_error_log_service.reset_mock()
    route_mock_error_log_service.process_get_error_logs.return_value = {"logs": [], "total": 0}
    route_mock_error_log_service.process_get_error_log_details.return_value = {}
    route_mock_error_log_service.process_find_error_log_by_info.return_value = {}
    route_mock_error_log_service.process_delete_error_logs_by_ids.return_value = 1
    route_mock_error_log_service.process_delete_all_error_logs.return_value = None
    route_mock_error_log_service.process_delete_error_log_by_id.return_value = True
    
    # Reset chat_service mock
    route_mock_chat_service.reset_mock()
    route_mock_chat_service.generate_content.return_value = {"candidates": [{"content": {"parts": [{"text": "Hello, world!"}]}}]}
    route_mock_chat_service.count_tokens.return_value = {"totalTokens": 123}
    
    # Reset embedding_service mock
    route_mock_embedding_service.reset_mock()
    route_mock_embedding_service.embed_content.return_value = {"embedding": {"values": [0.1, 0.2, 0.3]}}
    
    # Reset config_service mock
    route_mock_config_service.reset_mock()
    route_mock_config_service.get_config.return_value = {"LOG_LEVEL": "INFO", "API_KEYS": []}
    route_mock_config_service.update_config.return_value = {"status": "updated"}
    route_mock_config_service.reset_config.return_value = {"status": "reset"}
    route_mock_config_service.delete_key.return_value = {"success": True}
    route_mock_config_service.delete_selected_keys.return_value = {"success": True, "deleted_count": 0}
    route_mock_config_service.fetch_ui_models.return_value = {"models": []}
    
    # Reset proxy_check_service mock
    route_mock_proxy_check_service.reset_mock()
    import time
    route_mock_proxy_check_service.check_single_proxy.return_value = {
        "proxy": "proxy1",
        "is_available": True,
        "response_time": 0.5,
        "error_message": None,
        "checked_at": time.time(),
    }
    route_mock_proxy_check_service.check_multiple_proxies.return_value = []
    route_mock_proxy_check_service.get_cache_stats.return_value = {"hits": 0, "misses": 0}
    route_mock_proxy_check_service.clear_cache.return_value = None


@pytest.fixture(scope="function")
def route_test_app(
    route_mock_key_manager,
    route_mock_error_log_service,
    route_db_engine,  # Use route-specific db_engine
    route_async_db_engine,  # Use route-specific async db_engine for KeyManager
    route_mock_chat_service,
    route_mock_embedding_service,
    route_mock_config_service,
    route_mock_proxy_check_service,
):
    """
    Function-scoped fixture to create a test app for route tests.
    Each test gets a fresh app instance to avoid state leakage.
    """
    # route_db_engine and route_async_db_engine fixtures already set up the test databases
    app = create_app()

    # Store original overrides to restore them
    original_overrides = app.dependency_overrides.copy()

    async def override_get_key_manager():
        return route_mock_key_manager

    app.dependency_overrides[gemini_routes.get_key_manager] = override_get_key_manager
    app.dependency_overrides[openai_routes.get_key_manager] = override_get_key_manager
    app.dependency_overrides[
        vertex_express_routes.get_key_manager
    ] = override_get_key_manager
    app.dependency_overrides[
        openai_compatible_routes.get_key_manager
    ] = override_get_key_manager
    app.dependency_overrides[key_routes.get_key_manager] = override_get_key_manager

    async def override_get_error_log_service_dep():
        return route_mock_error_log_service

    app.dependency_overrides[
        get_error_log_service
    ] = override_get_error_log_service_dep

    async def override_get_chat_service():
        return route_mock_chat_service

    app.dependency_overrides[gemini_routes.get_chat_service] = override_get_chat_service

    async def override_get_embedding_service():
        return route_mock_embedding_service

    app.dependency_overrides[
        gemini_routes.get_embedding_service
    ] = override_get_embedding_service

    # Add ConfigService dependency override
    from app.dependencies import get_config_service
    from app.router import config_routes
    
    def override_get_config_service():
        return route_mock_config_service

    app.dependency_overrides[get_config_service] = override_get_config_service

    # Add ProxyCheckService dependency override
    from app.service.proxy.proxy_check_service import get_proxy_check_service
    
    def override_get_proxy_check_service():
        return route_mock_proxy_check_service

    app.dependency_overrides[get_proxy_check_service] = override_get_proxy_check_service

    async def mock_security_dependency():
        pass

    app.dependency_overrides[
        gemini_routes.security_service.verify_key_or_goog_api_key
    ] = mock_security_dependency

    app.dependency_overrides[
        openai_compatible_routes.security_service.verify_authorization
    ] = mock_security_dependency

    async def override_claude_proxy_service():
        return route_mock_chat_service

    app.dependency_overrides[claude_routes.ClaudeProxyService] = override_claude_proxy_service
    app.dependency_overrides[claude_routes.verify_auth_token] = mock_security_dependency

    yield app

    # Clean up dependency overrides after each test - restore to original state
    app.dependency_overrides.clear()
    app.dependency_overrides.update(original_overrides)


@pytest.fixture(scope="function")
def test_app(route_test_app):
    """
    Alias for route_test_app for backwards compatibility.
    Some tests use test_app instead of route_test_app.
    """
    return route_test_app


@pytest.fixture(scope="function")
def mock_key_manager(route_mock_key_manager):
    """
    Alias for route_mock_key_manager for backwards compatibility.
    Allows tests to use the same fixture name across different modules.
    """
    return route_mock_key_manager


@pytest.fixture(scope="function")
def mock_error_log_service(route_mock_error_log_service):
    """
    Alias for route_mock_error_log_service for backwards compatibility.
    Allows tests to use the same fixture name across different modules.
    """
    return route_mock_error_log_service


@pytest.fixture(scope="function")
def mock_chat_service(route_mock_chat_service):
    """
    Alias for route_mock_chat_service for backwards compatibility.
    Allows tests to use the same fixture name across different modules.
    """
    return route_mock_chat_service


@pytest.fixture(scope="function")
def mock_embedding_service(route_mock_embedding_service):
    """
    Alias for route_mock_embedding_service for backwards compatibility.
    Allows tests to use the same fixture name across different modules.
    """
    return route_mock_embedding_service


@pytest.fixture(scope="function")
def client(route_test_app):
    """
    Function-scoped fixture to create a TestClient for route tests.
    Each test gets a fresh client instance.
    """
    with TestClient(route_test_app) as test_client:
        yield test_client

