"""
Integration tests for scheduler routes.
"""

import pytest


@pytest.mark.asyncio
async def test_start_scheduler(test_client, auth_token):
    """Test starting scheduler."""
    cookies = {"auth_token": auth_token}
    response = test_client.post("/api/scheduler/start", cookies=cookies)
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


@pytest.mark.asyncio
async def test_stop_scheduler(test_client, auth_token):
    """Test stopping scheduler."""
    cookies = {"auth_token": auth_token}
    response = test_client.post("/api/scheduler/stop", cookies=cookies)
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


@pytest.mark.asyncio
async def test_start_scheduler_unauthorized(test_client):
    """Test starting scheduler without authentication."""
    response = test_client.post("/api/scheduler/start")
    
    assert response.status_code == 401

