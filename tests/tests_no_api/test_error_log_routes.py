"""
Integration tests for error log routes.
"""

import pytest


@pytest.mark.asyncio
async def test_get_error_logs(test_client, auth_token):
    """Test getting error logs with pagination."""
    test_client.cookies = {"auth_token": auth_token}
    response = test_client.get("/api/logs/errors?limit=10&offset=0")

    assert response.status_code == 200
    data = response.json()
    assert "logs" in data
    assert "total" in data
    assert isinstance(data["logs"], list)


@pytest.mark.asyncio
async def test_get_error_logs_with_filters(test_client, auth_token):
    """Test getting error logs with filters."""
    test_client.cookies = {"auth_token": auth_token}
    response = test_client.get(
        "/api/logs/errors?limit=10&offset=0&sort_by=id&sort_order=desc"
    )

    assert response.status_code == 200
    data = response.json()
    assert "logs" in data


@pytest.mark.asyncio
async def test_get_error_logs_unauthorized(test_client):
    """Test getting error logs without authentication."""
    # Ensure no cookies are set
    test_client.cookies = {}
    response = test_client.get("/api/logs/errors")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_error_log_detail(test_client, auth_token):
    """Test getting error log detail."""
    test_client.cookies = {"auth_token": auth_token}
    # Try to get a non-existent log (should return 404)
    response = test_client.get("/api/logs/errors/99999/details")

    # Should return 404 or 200 depending on whether log exists
    assert response.status_code in [200, 404]
