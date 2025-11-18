"""
Integration tests for key management routes.
"""

import pytest


@pytest.mark.asyncio
async def test_get_keys_paginated(test_client, auth_token):
    """Test getting paginated keys."""
    cookies = {"auth_token": auth_token}
    response = test_client.get("/api/keys?page=1&limit=10", cookies=cookies)
    
    assert response.status_code == 200
    data = response.json()
    assert "keys" in data
    assert "total_items" in data
    assert "current_page" in data


@pytest.mark.asyncio
async def test_get_keys_with_filters(test_client, auth_token):
    """Test getting keys with filters."""
    cookies = {"auth_token": auth_token}
    response = test_client.get(
        "/api/keys?page=1&limit=10&status=valid&search=test",
        cookies=cookies
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "keys" in data


@pytest.mark.asyncio
async def test_get_all_keys(test_client, auth_token):
    """Test getting all keys."""
    cookies = {"auth_token": auth_token}
    response = test_client.get("/api/keys/all", cookies=cookies)
    
    assert response.status_code == 200
    data = response.json()
    assert "valid_keys" in data
    assert "invalid_keys" in data
    assert "total_count" in data


@pytest.mark.asyncio
async def test_get_keys_unauthorized(test_client):
    """Test getting keys without authentication."""
    response = test_client.get("/api/keys")
    
    assert response.status_code == 401

