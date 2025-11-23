from unittest.mock import patch
import json
from datetime import datetime
import pytest

from app.config.config import settings


# Test for the /api/logs/errors endpoint (GET)
def test_get_error_logs_api_success(route_client, route_mock_error_log_service):
    """Test successful retrieval of error logs."""
    route_mock_error_log_service.process_get_error_logs.return_value = {
        "logs": [],
        "total": 0,
    }
    response = route_client.get(
        "/api/logs/errors", cookies={"auth_token": settings.AUTH_TOKEN}
    )
    assert response.status_code == 200
    res_json = response.json()
    assert "logs" in res_json
    assert "total" in res_json
    assert response.json() == {"logs": [], "total": 0}
    route_mock_error_log_service.process_get_error_logs.assert_awaited_once()
    get_logs_kwargs = (
        route_mock_error_log_service.process_get_error_logs.await_args.kwargs
    )
    assert "session" in get_logs_kwargs


@pytest.mark.no_mock_auth
@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
def test_get_error_logs_api_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to get_error_logs_api."""
    response = route_client.get(
        "/api/logs/errors", cookies={"auth_token": "invalid_token"}
    )
    assert response.status_code == 401


# Test for the /api/logs/errors/{log_id}/details endpoint (GET)
def test_get_error_log_detail_api_success(route_client, route_mock_error_log_service):
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
    response = route_client.get(
        "/api/logs/errors/1/details", cookies={"auth_token": settings.AUTH_TOKEN}
    )
    assert response.status_code == 200
    assert response.json() == log_detail
    route_mock_error_log_service.process_get_error_log_details.assert_awaited_once()
    detail_kwargs = (
        route_mock_error_log_service.process_get_error_log_details.await_args.kwargs
    )
    assert detail_kwargs["log_id"] == 1
    assert "session" in detail_kwargs


@pytest.mark.no_mock_auth
@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
def test_get_error_log_detail_api_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to get_error_log_detail_api."""
    response = route_client.get(
        "/api/logs/errors/1/details", cookies={"auth_token": "invalid_token"}
    )
    assert response.status_code == 401


# Test for the /api/logs/errors/lookup endpoint (GET)
def test_lookup_error_log_by_info_success(route_client, route_mock_error_log_service):
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
    route_mock_error_log_service.process_find_error_log_by_info.return_value = (
        log_detail
    )
    timestamp = datetime.utcnow()
    response = route_client.get(
        f"/api/logs/errors/lookup?gemini_key=test_key&timestamp={timestamp.isoformat()}",
        cookies={"auth_token": settings.AUTH_TOKEN},
    )
    assert response.status_code == 200
    assert response.json() == log_detail
    route_mock_error_log_service.process_find_error_log_by_info.assert_awaited_once()
    lookup_kwargs = (
        route_mock_error_log_service.process_find_error_log_by_info.await_args.kwargs
    )
    assert lookup_kwargs["gemini_key"] == "test_key"
    assert "session" in lookup_kwargs


@pytest.mark.no_mock_auth
@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
def test_lookup_error_log_by_info_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to lookup_error_log_by_info."""
    timestamp = datetime.utcnow().isoformat()
    response = route_client.get(
        f"/api/logs/errors/lookup?gemini_key=test_key&timestamp={timestamp}",
        cookies={"auth_token": "invalid_token"},
    )
    assert response.status_code == 401


# Test for the /api/logs/errors endpoint (DELETE)
def test_delete_error_logs_bulk_api_success(route_client, route_mock_error_log_service):
    """Test successful bulk deletion of error logs."""
    route_mock_error_log_service.process_delete_error_logs_by_ids.return_value = 1
    response = route_client.request(
        "DELETE",
        "/api/logs/errors",
        content=json.dumps({"ids": [1, 2]}),
        cookies={"auth_token": settings.AUTH_TOKEN},
    )
    assert response.status_code == 204
    route_mock_error_log_service.process_delete_error_logs_by_ids.assert_awaited_once()
    bulk_call = route_mock_error_log_service.process_delete_error_logs_by_ids.await_args
    assert bulk_call.args[0] == [1, 2]
    assert "session" in bulk_call.kwargs


@pytest.mark.no_mock_auth
@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
def test_delete_error_logs_bulk_api_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to delete_error_logs_bulk_api."""
    response = route_client.request(
        "DELETE",
        "/api/logs/errors",
        content=json.dumps({"ids": [1, 2]}),
        cookies={"auth_token": "invalid_token"},
    )
    assert response.status_code == 401


# Test for the /api/logs/errors/all endpoint (DELETE)
def test_delete_all_error_logs_api_success(route_client, route_mock_error_log_service):
    """Test successful deletion of all error logs."""
    response = route_client.delete(
        "/api/logs/errors/all", cookies={"auth_token": settings.AUTH_TOKEN}
    )
    assert response.status_code == 204
    route_mock_error_log_service.process_delete_all_error_logs.assert_awaited_once()
    delete_all_kwargs = (
        route_mock_error_log_service.process_delete_all_error_logs.await_args.kwargs
    )
    assert "session" in delete_all_kwargs


@pytest.mark.no_mock_auth
@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
def test_delete_all_error_logs_api_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to delete_all_error_logs_api."""
    response = route_client.delete(
        "/api/logs/errors/all",
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401


# Test for the /api/logs/errors/{log_id} endpoint (DELETE)
def test_delete_error_log_api_success(route_client, route_mock_error_log_service):
    """Test successful deletion of a single error log."""
    route_mock_error_log_service.process_delete_error_log_by_id.return_value = True
    response = route_client.delete(
        "/api/logs/errors/1", cookies={"auth_token": settings.AUTH_TOKEN}
    )
    assert response.status_code == 204
    route_mock_error_log_service.process_delete_error_log_by_id.assert_awaited_once()
    delete_call = route_mock_error_log_service.process_delete_error_log_by_id.await_args
    assert delete_call.args[0] == 1
    assert "session" in delete_call.kwargs


@pytest.mark.no_mock_auth
@patch("app.router.error_log_routes.verify_auth_token", return_value=False)
def test_delete_error_log_api_unauthorized(mock_verify_auth, route_client):
    """Test unauthorized access to delete_error_log_api."""
    response = route_client.delete(
        "/api/logs/errors/1",
        cookies={"auth_token": "invalid_token"},
        follow_redirects=False,
    )
    assert response.status_code == 401
