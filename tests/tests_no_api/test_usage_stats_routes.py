"""
Integration tests for usage statistics routes.
"""

import pytest


@pytest.mark.asyncio
async def test_get_usage_stats(test_client, auth_token):
    """Test getting usage statistics."""
    # /usage_stats/ is protected by AuthMiddleware, requires authentication
    cookies = {"auth_token": auth_token}
    response = test_client.get("/usage_stats/", cookies=cookies)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, (dict, list))

