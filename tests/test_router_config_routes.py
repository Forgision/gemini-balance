from unittest.mock import AsyncMock, patch, MagicMock
import pytest
import time
from app.core.application import create_app
from fastapi.testclient import TestClient
# from tests.conftest import mock_unauthorized_access
from app.service.proxy.proxy_check_service import ProxyCheckResult
from fastapi import HTTPException


@patch("app.middleware.middleware.verify_auth_token", return_value=True)
@patch("app.router.config_routes.verify_auth_token", return_value=True)
def test_get_config_success(mock_verify_route, mock_verify_middleware, client):# Test is working
    """Test successful retrieval of configuration."""
    response = client.get(
        "/api/config",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    config = response.json()
    assert isinstance(config, dict)
    assert config  # Asserts that the dictionary is not empty


def test_get_config_unauthorized(client): # Test is working
    """Test unauthorized access to get_config."""
    response = client.get(
        "/api/config",
        cookies={"auth_token": "invalid_token"}
    )
    # Redirect to home due to unauthorization
    assert response.url == "http://testserver/"

def test_update_config_success(client):
    """Test successful update of configuration."""
    mock_config_service = AsyncMock()
    mock_config_service.update_config.return_value = {"status": "updated"}

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    with patch("app.log.logger.Logger.update_log_levels") as mock_update_logs:
        response = client.put(
            "/api/config",
            json={"LOG_LEVEL": "DEBUG"},
            cookies={"auth_token": "test_auth_token"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "updated"}
    mock_config_service.update_config.assert_called_once_with({"LOG_LEVEL": "DEBUG"})
    mock_update_logs.assert_called_once_with("DEBUG")
    app.dependency_overrides.clear()

def test_update_config_unauthorized(client, mock_unauthorized_access):
    """Test unauthorized access to update_config."""
    response = client.put(
        "/api/config",
        json={"LOG_LEVEL": "DEBUG"},
        cookies={"auth_token": "invalid_token"},
        allow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/"

def test_update_config_error(client):
    """Test error handling when config update fails."""
    mock_config_service = AsyncMock()
    mock_config_service.update_config.side_effect = Exception("Update failed")

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.put(
        "/api/config",
        json={"LOG_LEVEL": "INFO"},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 400
    assert "Update failed" in response.text
    app.dependency_overrides.clear()

def test_reset_config_success(client):
    """Test successful reset of configuration."""
    mock_config_service = AsyncMock()
    mock_config_service.reset_config.return_value = {"status": "reset"}

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.post(
        "/api/config/reset",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "reset"}
    mock_config_service.reset_config.assert_called_once()
    app.dependency_overrides.clear()

def test_reset_config_unauthorized(client, mock_unauthorized_access):
    """Test unauthorized access to reset_config."""
    response = client.post(
        "/api/config/reset",
        cookies={"auth_token": "invalid_token"},
        allow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/"

def test_reset_config_error(client):
    """Test error handling during config reset."""
    mock_config_service = AsyncMock()
    mock_config_service.reset_config.side_effect = Exception("Reset failed")

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.post(
        "/api/config/reset",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 400
    assert "Reset failed" in response.text
    app.dependency_overrides.clear()

# Tests for key deletion
def test_delete_single_key_success(client):
    """Test successful deletion of a single key."""
    mock_config_service = AsyncMock()
    mock_config_service.delete_key.return_value = {"success": True}

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.delete(
        "/api/config/keys/test_key",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True}
    mock_config_service.delete_key.assert_called_once_with("test_key")
    app.dependency_overrides.clear()

def test_delete_single_key_unauthorized(client, mock_unauthorized_access):
    """Test unauthorized access to delete_single_key."""
    response = client.delete(
        "/api/config/keys/test_key",
        cookies={"auth_token": "invalid_token"},
        allow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/"

def test_delete_single_key_not_found(client):
    """Test deleting a key that is not found."""
    mock_config_service = AsyncMock()
    mock_config_service.delete_key.return_value = {"success": False, "message": "Key not found"}

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.delete(
        "/api/config/keys/non_existent_key",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 404
    assert "Key not found" in response.text
    app.dependency_overrides.clear()

def test_delete_selected_keys_success(client):
    """Test successful deletion of selected keys."""
    mock_config_service = AsyncMock()
    mock_config_service.delete_selected_keys.return_value = {"success": True, "deleted_count": 2}

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.post(
        "/api/config/keys/delete-selected",
        json={"keys": ["key1", "key2"]},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "deleted_count": 2}
    mock_config_service.delete_selected_keys.assert_called_once_with(["key1", "key2"])
    app.dependency_overrides.clear()

def test_delete_selected_keys_unauthorized(client, mock_unauthorized_access):
    """Test unauthorized access to delete_selected_keys."""
    response = client.post(
        "/api/config/keys/delete-selected",
        json={"keys": ["key1", "key2"]},
        cookies={"auth_token": "invalid_token"},
        allow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/"

def test_delete_selected_keys_no_keys_provided(client):
    """Test request to delete selected keys with an empty list."""
    response = client.post(
        "/api/config/keys/delete-selected",
        json={"keys": []},
        cookies={"auth_token": "test_auth_token"},
    )
    assert response.status_code == 400
    assert "No keys provided" in response.text

# Tests for proxy endpoints
def test_check_single_proxy_success(client):
    """Test successful check of a single proxy."""
    mock_proxy_service = AsyncMock()
    mock_proxy_service.check_single_proxy.return_value = {
        "proxy": "proxy1",
        "is_available": True,
        "response_time": 0.5,
        "error_message": None,
        "checked_at": time.time(),
    }

    with patch("app.router.config_routes.get_proxy_check_service", return_value=mock_proxy_service):
        response = client.post(
            "/api/config/proxy/check",
            json={"proxy": "proxy1"},
            cookies={"auth_token": "test_auth_token"},
        )

    assert response.status_code == 200
    assert response.json()["proxy"] == "proxy1"
    mock_proxy_service.check_single_proxy.assert_called_once_with("proxy1", True)

def test_check_single_proxy_unauthorized(client, mock_unauthorized_access):
    """Test unauthorized access to check_single_proxy."""
    response = client.post(
        "/api/config/proxy/check",
        json={"proxy": "proxy1"},
        cookies={"auth_token": "invalid_token"},
        allow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/"

def test_check_all_proxies_success(client):
    """Test successful check of multiple proxies."""
    mock_proxy_service = AsyncMock()
    mock_proxy_service.check_multiple_proxies.return_value = [
        {
            "proxy": "proxy1",
            "is_available": True,
            "response_time": 0.5,
            "error_message": None,
            "checked_at": time.time(),
        }
    ]

    with patch("app.router.config_routes.get_proxy_check_service", return_value=mock_proxy_service):
        response = client.post(
            "/api/config/proxy/check-all",
            json={"proxies": ["proxy1"]},
            cookies={"auth_token": "test_auth_token"},
        )

    assert response.status_code == 200
    assert response.json()[0]["proxy"] == "proxy1"
    mock_proxy_service.check_multiple_proxies.assert_called_once()

def test_check_all_proxies_unauthorized(client, mock_unauthorized_access):
    """Test unauthorized access to check_all_proxies."""
    response = client.post(
        "/api/config/proxy/check-all",
        json={"proxies": ["proxy1"]},
        cookies={"auth_token": "invalid_token"},
        allow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/"

def test_get_proxy_cache_stats_success(client):
    """Test successful retrieval of proxy cache stats."""
    mock_proxy_service = AsyncMock()
    mock_proxy_service.get_cache_stats.return_value = {"hits": 10, "misses": 5}

    with patch("app.router.config_routes.get_proxy_check_service", return_value=mock_proxy_service):
        response = client.get(
            "/api/config/proxy/cache-stats",
            cookies={"auth_token": "test_auth_token"},
        )

    assert response.status_code == 200
    assert response.json() == {"hits": 10, "misses": 5}
    mock_proxy_service.get_cache_stats.assert_called_once()

def test_get_proxy_cache_stats_unauthorized(client, mock_unauthorized_access):
    """Test unauthorized access to get_proxy_cache_stats."""
    response = client.get(
        "/api/config/proxy/cache-stats",
        cookies={"auth_token": "invalid_token"},
        allow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/"

def test_clear_proxy_cache_success(client):
    """Test successful clearing of proxy cache."""
    mock_proxy_service = AsyncMock()
    mock_proxy_service.clear_cache.return_value = None

    with patch("app.router.config_routes.get_proxy_check_service", return_value=mock_proxy_service):
        response = client.post(
            "/api/config/proxy/clear-cache",
            cookies={"auth_token": "test_auth_token"},
        )

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "Proxy check cache cleared"}
    mock_proxy_service.clear_cache.assert_called_once()

def test_clear_proxy_cache_unauthorized(client, mock_unauthorized_access):
    """Test unauthorized access to clear_proxy_cache."""
    response = client.post(
        "/api/config/proxy/clear-cache",
        cookies={"auth_token": "invalid_token"},
        allow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/"

# Tests for UI models endpoint
def test_get_ui_models_success(client):
    """Test successful retrieval of UI models."""
    mock_config_service = AsyncMock()
    mock_config_service.fetch_ui_models.return_value = {"models": ["model1", "model2"]}

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.get(
        "/api/config/ui/models",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    assert response.json() == {"models": ["model1", "model2"]}
    mock_config_service.fetch_ui_models.assert_called_once()
    app.dependency_overrides.clear()

def test_get_ui_models_unauthorized(client, mock_unauthorized_access):
    """Test unauthorized access to get_ui_models."""
    response = client.get(
        "/api/config/ui/models",
        cookies={"auth_token": "invalid_token"},
        allow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/"

def test_delete_single_key_error(client):
    """Test error handling when deleting a single key fails."""
    mock_config_service = AsyncMock()
    mock_config_service.delete_key.side_effect = Exception("Deletion failed")

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.delete(
        "/api/config/keys/test_key",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 500
    assert "Error deleting key: Deletion failed" in response.text
    app.dependency_overrides.clear()

def test_delete_single_key_generic_error(client):
    """Test error handling when deleting a single key fails with a generic error (not 404)."""
    mock_config_service = AsyncMock()
    mock_config_service.delete_key.return_value = {"success": False, "message": "Generic deletion error"}

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.delete(
        "/api/config/keys/test_key",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 400
    assert "Generic deletion error" in response.text
    mock_config_service.delete_key.assert_called_once_with("test_key")
    app.dependency_overrides.clear()

def test_delete_selected_keys_error(client):
    """Test error handling when bulk deleting keys fails."""
    mock_config_service = AsyncMock()
    mock_config_service.delete_selected_keys.side_effect = Exception("Bulk deletion failed")

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.post(
        "/api/config/keys/delete-selected",
        json={"keys": ["key1", "key2"]},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 500
    assert "Error bulk deleting keys: Bulk deletion failed" in response.text
    app.dependency_overrides.clear()

def test_check_single_proxy_error(client):
    """Test error handling when checking a single proxy fails."""
    mock_proxy_service = AsyncMock()
    mock_proxy_service.check_single_proxy.side_effect = Exception("Proxy check error")

    with patch("app.router.config_routes.get_proxy_check_service", return_value=mock_proxy_service):
        response = client.post(
            "/api/config/proxy/check",
            json={"proxy": "proxy1"},
            cookies={"auth_token": "test_auth_token"},
        )

    assert response.status_code == 500
    assert "Proxy check failed: Proxy check error" in response.text

def test_check_all_proxies_error(client):
    """Test error handling when checking multiple proxies fails."""
    mock_proxy_service = AsyncMock()
    mock_proxy_service.check_multiple_proxies.side_effect = Exception("Batch proxy check error")

    with patch("app.router.config_routes.get_proxy_check_service", return_value=mock_proxy_service):
        response = client.post(
            "/api/config/proxy/check-all",
            json={"proxies": ["proxy1"]},
            cookies={"auth_token": "test_auth_token"},
        )

    assert response.status_code == 500
    assert "Batch proxy check failed: Batch proxy check error" in response.text

def test_get_proxy_cache_stats_error(client):
    """Test error handling when retrieving proxy cache stats fails."""
    mock_proxy_service = AsyncMock()
    mock_proxy_service.get_cache_stats.side_effect = Exception("Cache stats error")

    with patch("app.router.config_routes.get_proxy_check_service", return_value=mock_proxy_service):
        response = client.get(
            "/api/config/proxy/cache-stats",
            cookies={"auth_token": "test_auth_token"},
        )

    assert response.status_code == 500
    assert "Get cache stats failed: Cache stats error" in response.text

def test_clear_proxy_cache_error(client):
    """Test error handling when clearing proxy cache fails."""
    mock_proxy_service = AsyncMock()
    mock_proxy_service.clear_cache.side_effect = Exception("Clear cache error")

    with patch("app.router.config_routes.get_proxy_check_service", return_value=mock_proxy_service):
        response = client.post(
            "/api/config/proxy/clear-cache",
            cookies={"auth_token": "test_auth_token"},
        )

    assert response.status_code == 500
    assert "Clear cache failed: Clear cache error" in response.text

def test_get_ui_models_error(client):
    """Test error handling when retrieving UI models fails."""
    mock_config_service = AsyncMock()
    mock_config_service.fetch_ui_models.side_effect = Exception("Fetch UI models error")

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.get(
        "/api/config/ui/models",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 500
    assert "An unexpected error occurred while fetching UI models: Fetch UI models error" in response.text
    app.dependency_overrides.clear()

def test_delete_selected_keys_partial_failure(client):
    """Test partial failure in deleting selected keys."""
    mock_config_service = AsyncMock()
    mock_config_service.delete_selected_keys.return_value = {"success": False, "deleted_count": 0, "message": "Some keys not found."}

    from app.dependencies import get_config_service
    app = client.app
    app.dependency_overrides[get_config_service] = lambda: mock_config_service

    response = client.post(
        "/api/config/keys/delete-selected",
        json={"keys": ["key1", "key2"]},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 400
    assert "Some keys not found." in response.text
    app.dependency_overrides.clear()
