"""
Integration tests for statistics routes.
"""

import pytest
from tests.tests_no_api.conftest import TEST_API_KEYS


@pytest.mark.asyncio
async def test_get_keys_status_page(test_client, auth_token):
    """Test getting keys status page (HTML)."""
    cookies = {"auth_token": auth_token}
    # The route is /keys, not /keys_status
    response = test_client.get("/keys", cookies=cookies)
    
    # Should return HTML or redirect
    assert response.status_code in [200, 302]
    if response.status_code == 200:
        # Check if it's HTML
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type or "html" in content_type.lower()


@pytest.mark.asyncio
async def test_get_key_usage_details(test_client, auth_token, test_key_manager):
    """Test getting key usage details."""
    cookies = {"auth_token": auth_token}
    
    # Use a test API key
    api_key = TEST_API_KEYS[0] if TEST_API_KEYS else "test_key"
    
    response = test_client.get(f"/api/key-usage-details/{api_key}", cookies=cookies)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_get_key_usage_details_unauthorized(test_client):
    """Test getting key usage details without authentication."""
    response = test_client.get("/api/key-usage-details/test_key")
    
    assert response.status_code == 401

