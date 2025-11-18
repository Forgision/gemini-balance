"""
Integration tests for main routes (root, health, etc.).
"""

import pytest


@pytest.mark.asyncio
async def test_root_endpoint(test_client):
    """Test root endpoint."""
    response = test_client.get("/")
    
    # Root endpoint may redirect or return HTML
    assert response.status_code in [200, 302, 307]


@pytest.mark.asyncio
async def test_health_endpoint(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    
    assert response.status_code == 200
    # Health endpoint should return status
    data = response.json() if response.headers.get("content-type") == "application/json" else None
    if data:
        assert "status" in data or "health" in data

