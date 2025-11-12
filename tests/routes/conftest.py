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


@pytest.fixture(scope="function")
def route_test_app(
    mock_key_manager,
    mock_error_log_service,
    route_db_engine,  # Use route-specific db_engine
    mock_chat_service,
    mock_embedding_service,
):
    """
    Function-scoped fixture to create a test app for route tests.
    Each test gets a fresh app instance to avoid state leakage.
    """
    # route_db_engine fixture already sets up the test database
    app = create_app()

    async def override_get_key_manager():
        return mock_key_manager

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
        return mock_error_log_service

    app.dependency_overrides[
        get_error_log_service
    ] = override_get_error_log_service_dep

    async def override_get_chat_service():
        return mock_chat_service

    app.dependency_overrides[gemini_routes.get_chat_service] = override_get_chat_service

    async def override_get_embedding_service():
        return mock_embedding_service

    app.dependency_overrides[
        gemini_routes.get_embedding_service
    ] = override_get_embedding_service

    async def mock_security_dependency():
        pass

    app.dependency_overrides[
        gemini_routes.security_service.verify_key_or_goog_api_key
    ] = mock_security_dependency

    app.dependency_overrides[
        openai_compatible_routes.security_service.verify_authorization
    ] = mock_security_dependency

    async def override_claude_proxy_service():
        return mock_chat_service

    app.dependency_overrides[claude_routes.ClaudeProxyService] = override_claude_proxy_service
    app.dependency_overrides[claude_routes.verify_auth_token] = mock_security_dependency

    yield app

    # Clean up dependency overrides after each test
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(route_test_app):
    """
    Function-scoped fixture to create a TestClient for route tests.
    Each test gets a fresh client instance.
    """
    with TestClient(route_test_app) as test_client:
        yield test_client

