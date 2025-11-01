from unittest.mock import AsyncMock, patch, MagicMock
import pytest
import time
from app.core.application import create_app
from fastapi.testclient import TestClient
from app.service.proxy.proxy_check_service import ProxyCheckResult
from fastapi import HTTPException


@patch("app.middleware.middleware.verify_auth_token", return_value=True)
@patch("app.router.config_routes.verify_auth_token", return_value=True)
def test_get_config_success(mock_verify_route, mock_verify_middleware, client):
    """Test successful retrieval of configuration."""
    response = client.get(
        "/api/config",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    config = response.json()
    assert isinstance(config, dict)
    assert config


@pytest.mark.no_mock_auth
@patch("app.middleware.middleware.verify_auth_token", return_value=False)
@patch("app.router.config_routes.verify_auth_token", return_value=False)
def test_get_config_unauthorized(mock_router_auth, mock_middleware_auth, client):
    """Test unauthorized access to get_config."""
    response = client.get(
        "/api/config",
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401


@patch("app.service.config.config_service.ConfigService.update_config")
def test_update_config_success(mock_update_config, client):
    """Test successful update of configuration."""
    mock_update_config.return_value = {"status": "updated"}

    with patch("app.log.logger.Logger.update_log_levels") as mock_update_logs:
        response = client.put(
            "/api/config",
            json={"LOG_LEVEL": "DEBUG"},
            cookies={"auth_token": "test_auth_token"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "updated"}
    mock_update_config.assert_called_once_with({"LOG_LEVEL": "DEBUG"})
    mock_update_logs.assert_called_once_with("DEBUG")


@pytest.mark.no_mock_auth
@patch("app.middleware.middleware.verify_auth_token", return_value=False)
@patch("app.router.config_routes.verify_auth_token", return_value=False)
def test_update_config_unauthorized(mock_router_auth, mock_middleware_auth, client):
    """Test unauthorized access to update_config."""
    response = client.put(
        "/api/config",
        json={"LOG_LEVEL": "DEBUG"},
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401


@patch("app.service.config.config_service.ConfigService.update_config")
def test_update_config_error(mock_update_config, client):
    """Test error handling when config update fails."""
    mock_update_config.side_effect = Exception("Update failed")

    response = client.put(
        "/api/config",
        json={"LOG_LEVEL": "INFO"},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 400
    assert "Update failed" in response.text


@patch("app.service.config.config_service.ConfigService.reset_config")
def test_reset_config_success(mock_reset_config, client):
    """Test successful reset of configuration."""
    mock_reset_config.return_value = {"status": "reset"}

    response = client.post(
        "/api/config/reset",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "reset"}
    mock_reset_config.assert_called_once()


@pytest.mark.no_mock_auth
@patch("app.middleware.middleware.verify_auth_token", return_value=False)
@patch("app.router.config_routes.verify_auth_token", return_value=False)
def test_reset_config_unauthorized(mock_router_auth, mock_middleware_auth, client):
    """Test unauthorized access to reset_config."""
    response = client.post(
        "/api/config/reset",
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401


@patch("app.service.config.config_service.ConfigService.reset_config")
def test_reset_config_error(mock_reset_config, client):
    """Test error handling during config reset."""
    mock_reset_config.side_effect = Exception("Reset failed")

    response = client.post(
        "/api/config/reset",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 400
    assert "Reset failed" in response.text


# Tests for key deletion
@patch("app.service.config.config_service.ConfigService.delete_key")
def test_delete_single_key_success(mock_delete_key, client):
    """Test successful deletion of a single key."""
    mock_delete_key.return_value = {"success": True}

    response = client.delete(
        "/api/config/keys/test_key",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True}
    mock_delete_key.assert_called_once_with("test_key")


@pytest.mark.no_mock_auth
@patch("app.middleware.middleware.verify_auth_token", return_value=False)
@patch("app.router.config_routes.verify_auth_token", return_value=False)
def test_delete_single_key_unauthorized(mock_router_auth, mock_middleware_auth, client):
    """Test unauthorized access to delete_single_key."""
    response = client.delete(
        "/api/config/keys/test_key",
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401


@patch("app.service.config.config_service.ConfigService.delete_key")
def test_delete_single_key_not_found(mock_delete_key, client):
    """Test deleting a key that is not found."""
    mock_delete_key.return_value = {"success": False, "message": "Key not found"}

    response = client.delete(
        "/api/config/keys/non_existent_key",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 404
    assert "Key not found" in response.text


@patch("app.service.config.config_service.ConfigService.delete_selected_keys")
def test_delete_selected_keys_success(mock_delete_selected_keys, client):
    """Test successful deletion of selected keys."""
    mock_delete_selected_keys.return_value = {"success": True, "deleted_count": 2}

    response = client.post(
        "/api/config/keys/delete-selected",
        json={"keys": ["key1", "key2"]},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "deleted_count": 2}
    mock_delete_selected_keys.assert_called_once_with(["key1", "key2"])


@pytest.mark.no_mock_auth
@patch("app.middleware.middleware.verify_auth_token", return_value=False)
@patch("app.router.config_routes.verify_auth_token", return_value=False)
def test_delete_selected_keys_unauthorized(mock_router_auth, mock_middleware_auth, client):
    """Test unauthorized access to delete_selected_keys."""
    response = client.post(
        "/api/config/keys/delete-selected",
        json={"keys": ["key1", "key2"]},
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401


@patch("app.service.config.config_service.ConfigService.delete_selected_keys")
def test_delete_selected_keys_no_keys_provided(mock_delete_selected_keys, client):
    """Test request to delete selected keys with an empty list."""
    response = client.post(
        "/api/config/keys/delete-selected",
        json={"keys": []},
        cookies={"auth_token": "test_auth_token"},
    )
    assert response.status_code == 400
    assert "No keys provided" in response.text
    mock_delete_selected_keys.assert_not_called()


# Tests for proxy endpoints
@patch("app.service.proxy.proxy_check_service.ProxyCheckService.check_single_proxy")
def test_check_single_proxy_success(mock_check_single_proxy, client):
    """Test successful check of a single proxy."""
    mock_check_single_proxy.return_value = {
        "proxy": "proxy1",
        "is_available": True,
        "response_time": 0.5,
        "error_message": None,
        "checked_at": time.time(),
    }

    response = client.post(
        "/api/config/proxy/check",
        json={"proxy": "proxy1"},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    assert response.json()["proxy"] == "proxy1"
    mock_check_single_proxy.assert_called_once_with("proxy1", True)


@pytest.mark.no_mock_auth
@patch("app.middleware.middleware.verify_auth_token", return_value=False)
@patch("app.router.config_routes.verify_auth_token", return_value=False)
def test_check_single_proxy_unauthorized(mock_router_auth, mock_middleware_auth, client):
    """Test unauthorized access to check_single_proxy."""
    response = client.post(
        "/api/config/proxy/check",
        json={"proxy": "proxy1"},
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401


@patch("app.service.proxy.proxy_check_service.ProxyCheckService.check_multiple_proxies")
def test_check_all_proxies_success(mock_check_multiple_proxies, client):
    """Test successful check of multiple proxies."""
    mock_check_multiple_proxies.return_value = [
        {
            "proxy": "proxy1",
            "is_available": True,
            "response_time": 0.5,
            "error_message": None,
            "checked_at": time.time(),
        }
    ]

    response = client.post(
        "/api/config/proxy/check-all",
        json={"proxies": ["proxy1"]},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    res_json = response.json()
    assert isinstance(res_json, list)
    assert len(res_json) == 1
    assert response.json()[0]["proxy"] == "proxy1"
    mock_check_multiple_proxies.assert_called_once()


@pytest.mark.no_mock_auth
@patch("app.middleware.middleware.verify_auth_token", return_value=False)
@patch("app.router.config_routes.verify_auth_token", return_value=False)
def test_check_all_proxies_unauthorized(mock_router_auth, mock_middleware_auth, client):
    """Test unauthorized access to check_all_proxies."""
    response = client.post(
        "/api/config/proxy/check-all",
        json={"proxies": ["proxy1"]},
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401


@patch("app.service.proxy.proxy_check_service.ProxyCheckService.get_cache_stats")
def test_get_proxy_cache_stats_success(mock_get_cache_stats, client):
    """Test successful retrieval of proxy cache stats."""
    mock_get_cache_stats.return_value = {"hits": 10, "misses": 5}

    response = client.get(
        "/api/config/proxy/cache-stats",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    assert response.json() == {"hits": 10, "misses": 5}
    mock_get_cache_stats.assert_called_once()


@pytest.mark.no_mock_auth
@patch("app.middleware.middleware.verify_auth_token", return_value=False)
@patch("app.router.config_routes.verify_auth_token", return_value=False)
def test_get_proxy_cache_stats_unauthorized(mock_router_auth, mock_middleware_auth, client):
    """Test unauthorized access to get_proxy_cache_stats."""
    response = client.get(
        "/api/config/proxy/cache-stats",
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401


@patch("app.service.proxy.proxy_check_service.ProxyCheckService.clear_cache", new_callable=MagicMock)
def test_clear_proxy_cache_success(mock_clear_cache, client):
    """Test successful clearing of proxy cache."""
    mock_clear_cache.return_value = None

    response = client.post(
        "/api/config/proxy/clear-cache",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "Proxy check cache cleared"}
    mock_clear_cache.assert_called_once()


@pytest.mark.no_mock_auth
@patch("app.middleware.middleware.verify_auth_token", return_value=False)
@patch("app.router.config_routes.verify_auth_token", return_value=False)
def test_clear_proxy_cache_unauthorized(mock_router_auth, mock_middleware_auth, client):
    """Test unauthorized access to clear_proxy_cache."""
    response = client.post(
        "/api/config/proxy/clear-cache",
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401


# Tests for UI models endpoint
@patch("app.service.config.config_service.ConfigService.fetch_ui_models")
def test_get_ui_models_success(mock_fetch_ui_models, client):
    """Test successful retrieval of UI models."""
    mock_fetch_ui_models.return_value = {"models": ["model1", "model2"]}

    response = client.get(
        "/api/config/ui/models",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    assert response.json() == {"models": ["model1", "model2"]}
    mock_fetch_ui_models.assert_called_once()


@pytest.mark.no_mock_auth
@patch("app.middleware.middleware.verify_auth_token", return_value=False)
@patch("app.router.config_routes.verify_auth_token", return_value=False)
def test_get_ui_models_unauthorized(mock_router_auth, mock_middleware_auth, client):
    """Test unauthorized access to get_ui_models."""
    response = client.get(
        "/api/config/ui/models",
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401


@patch("app.service.config.config_service.ConfigService.delete_key")
def test_delete_single_key_error(mock_delete_key, client):
    """Test error handling when deleting a single key fails."""
    mock_delete_key.side_effect = Exception("Deletion failed")

    response = client.delete(
        "/api/config/keys/test_key",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 500
    assert "Error deleting key: Deletion failed" in response.text


@patch("app.service.config.config_service.ConfigService.delete_key")
def test_delete_single_key_generic_error(mock_delete_key, client):
    """Test error handling when deleting a single key fails with a generic error (not 404)."""
    mock_delete_key.return_value = {"success": False, "message": "Generic deletion error"}

    response = client.delete(
        "/api/config/keys/test_key",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 400
    assert "Generic deletion error" in response.text
    mock_delete_key.assert_called_once_with("test_key")


@patch("app.service.config.config_service.ConfigService.delete_selected_keys")
def test_delete_selected_keys_error(mock_delete_selected_keys, client):
    """Test error handling when bulk deleting keys fails."""
    mock_delete_selected_keys.side_effect = Exception("Bulk deletion failed")

    response = client.post(
        "/api/config/keys/delete-selected",
        json={"keys": ["key1", "key2"]},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 500
    assert "Error bulk deleting keys: Bulk deletion failed" in response.text


@patch("app.service.proxy.proxy_check_service.ProxyCheckService.check_single_proxy")
def test_check_single_proxy_error(mock_check_single_proxy, client):
    """Test error handling when checking a single proxy fails."""
    mock_check_single_proxy.side_effect = Exception("Proxy check error")

    response = client.post(
        "/api/config/proxy/check",
        json={"proxy": "proxy1"},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 500
    assert "Proxy check failed: Proxy check error" in response.text


@patch("app.service.proxy.proxy_check_service.ProxyCheckService.check_multiple_proxies")
def test_check_all_proxies_error(mock_check_multiple_proxies, client):
    """Test error handling when checking multiple proxies fails."""
    mock_check_multiple_proxies.side_effect = Exception("Batch proxy check error")

    response = client.post(
        "/api/config/proxy/check-all",
        json={"proxies": ["proxy1"]},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 500
    assert "Batch proxy check failed: Batch proxy check error" in response.text


@patch("app.service.proxy.proxy_check_service.ProxyCheckService.get_cache_stats")
def test_get_proxy_cache_stats_error(mock_get_cache_stats, client):
    """Test error handling when retrieving proxy cache stats fails."""
    mock_get_cache_stats.side_effect = Exception("Cache stats error")

    response = client.get(
        "/api/config/proxy/cache-stats",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 500
    assert "Get cache stats failed: Cache stats error" in response.text


@patch("app.service.proxy.proxy_check_service.ProxyCheckService.clear_cache")
def test_clear_proxy_cache_error(mock_clear_cache, client):
    """Test error handling when clearing proxy cache fails."""
    mock_clear_cache.side_effect = Exception("Clear cache error")

    response = client.post(
        "/api/config/proxy/clear-cache",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 500
    assert "Clear cache failed: Clear cache error" in response.text


@patch("app.service.config.config_service.ConfigService.fetch_ui_models")
def test_get_ui_models_error(mock_fetch_ui_models, client):
    """Test error handling when retrieving UI models fails."""
    mock_fetch_ui_models.side_effect = Exception("Fetch UI models error")

    response = client.get(
        "/api/config/ui/models",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 500
    assert "An unexpected error occurred while fetching UI models: Fetch UI models error" in response.text


@patch("app.service.config.config_service.ConfigService.delete_selected_keys")
def test_delete_selected_keys_partial_failure(mock_delete_selected_keys, client):
    """Test partial failure in deleting selected keys."""
    mock_delete_selected_keys.return_value = {"success": False, "deleted_count": 0, "message": "Some keys not found."}

    response = client.post(
        "/api/config/keys/delete-selected",
        json={"keys": ["key1", "key2"]},
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 400
    assert "Some keys not found." in response.text