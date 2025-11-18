"""
Integration tests for version routes.
"""

import pytest


@pytest.mark.asyncio
async def test_get_version(test_client):
    """Test getting version information."""
    # Use the correct route /api/version/check (not /version)
    response = test_client.get("/api/version/check")
    
    assert response.status_code == 200
    data = response.json()
    # Version endpoint should return version info
    assert "current_version" in data
    assert isinstance(data, dict)

