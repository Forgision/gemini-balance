from unittest.mock import AsyncMock, patch
import pytest
import json
from app.main import app
from datetime import datetime, timedelta


# Test for the /api/logs/errors endpoint (GET)
async def test_get_error_logs_api_success(mock_verify_auth_token, route_client, route_mock_error_log_service):
    """Test successful retrieval of error logs."""
    route_mock_error_log_service.process_get_error_logs.return_value = {"logs": [], "total": 0}
    response = route_client.get("/api/logs/errors", cookies={"auth_token": "test_token"})
    assert response.status_code == 200
    res_json = response.json()
    assert "logs" in res_json
    assert "total" in res_json
    assert response.json() == {"logs": [], "total": 0}
    route_mock_error_log_service.process_get_error_logs.assert_called_once()


@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
async def test_get_error_logs_api_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to get_error_logs_api."""
    response = route_client.get("/api/logs/errors", cookies={"auth_token": "invalid_token"})
    assert response.status_code == 401


# Test for the /api/logs/errors/{log_id}/details endpoint (GET)
async def test_get_error_log_detail_api_success(mock_verify_auth_token, route_client, route_mock_error_log_service):
    """Test successful retrieval of an error log's details."""
    log_detail = {
        "id": 1,
        "gemini_key": "test_key",
        "error_type": "test_error",
        "error_log": "details",
        "request_msg": "test_request",
        "model_name": "test_model",
        "request_time": datetime.utcnow().isoformat(),
        "error_code": 500,
    }
    route_mock_error_log_service.process_get_error_log_details.return_value = log_detail
    response = route_client.get("/api/logs/errors/1/details", cookies={"auth_token": "test_token"})
    assert response.status_code == 200
    assert response.json() == log_detail
    route_mock_error_log_service.process_get_error_log_details.assert_called_once_with(log_id=1)


@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
async def test_get_error_log_detail_api_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to get_error_log_detail_api."""
    response = route_client.get("/api/logs/errors/1/details", cookies={"auth_token": "invalid_token"})
    assert response.status_code == 401


# Test for the /api/logs/errors/lookup endpoint (GET)
async def test_lookup_error_log_by_info_success(mock_verify_auth_token, route_client, route_mock_error_log_service):
    """Test successful lookup of an error log."""
    log_detail = {
        "id": 1,
        "gemini_key": "test_key",
        "error_type": "test_error",
        "error_log": "details",
        "request_msg": "test_request",
        "model_name": "test_model",
        "request_time": datetime.utcnow().isoformat(),
        "error_code": 500,
    }
    route_mock_error_log_service.process_find_error_log_by_info.return_value = log_detail
    timestamp = datetime.utcnow()
    response = route_client.get(
        f"/api/logs/errors/lookup?gemini_key=test_key&timestamp={timestamp.isoformat()}",
        cookies={"auth_token": "test_token"},
    )
    assert response.status_code == 200
    assert response.json() == log_detail
    route_mock_error_log_service.process_find_error_log_by_info.assert_called_once()


@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
async def test_lookup_error_log_by_info_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to lookup_error_log_by_info."""
    timestamp = datetime.utcnow().isoformat()
    response = route_client.get(
        f"/api/logs/errors/lookup?gemini_key=test_key&timestamp={timestamp}",
        cookies={"auth_token": "invalid_token"},
    )
    assert response.status_code == 401


# Test for the /api/logs/errors endpoint (DELETE)
async def test_delete_error_logs_bulk_api_success(mock_verify_auth_token, route_client, route_mock_error_log_service):
    """Test successful bulk deletion of error logs."""
    route_mock_error_log_service.process_delete_error_logs_by_ids.return_value = 1
    response = route_client.request("DELETE", "/api/logs/errors", content=json.dumps({"ids": [1, 2]}), cookies={"auth_token": "test_token"})
    assert response.status_code == 204
    route_mock_error_log_service.process_delete_error_logs_by_ids.assert_called_once_with([1, 2])


@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
async def test_delete_error_logs_bulk_api_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to delete_error_logs_bulk_api."""
    response = route_client.request(
        "DELETE",
        "/api/logs/errors",
        content=json.dumps({"ids": [1, 2]}),
        cookies={"auth_token": "invalid_token"},
    )
    assert response.status_code == 401


# Test for the /api/logs/errors/all endpoint (DELETE)
async def test_delete_all_error_logs_api_success(mock_verify_auth_token, route_client, route_mock_error_log_service):
    """Test successful deletion of all error logs."""
    response = route_client.delete("/api/logs/errors/all", cookies={"auth_token": "test_token"})
    assert response.status_code == 204
    route_mock_error_log_service.process_delete_all_error_logs.assert_called_once()


@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
async def test_delete_all_error_logs_api_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to delete_all_error_logs_api."""
    response = route_client.delete("/api/logs/errors/all", cookies={"auth_token": "invalid_token"}, follow_redirects=False)
    assert response.status_code == 401


# Test for the /api/logs/errors/{log_id} endpoint (DELETE)
async def test_delete_error_log_api_success(mock_verify_auth_token, route_client, route_mock_error_log_service):
    """Test successful deletion of a single error log."""
    route_mock_error_log_service.process_delete_error_log_by_id.return_value = True
    response = route_client.delete("/api/logs/errors/1", cookies={"auth_token": "test_token"})
    assert response.status_code == 204
    route_mock_error_log_service.process_delete_error_log_by_id.assert_called_once_with(1)


@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
async def test_delete_error_log_api_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to delete_error_log_api."""
    response = route_client.delete("/api/logs/errors/1", cookies={"auth_token": "invalid_token"}, follow_redirects=False)
    assert response.status_code == 401