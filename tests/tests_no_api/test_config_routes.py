"""
Integration tests for configuration routes.
"""

import pytest

from app.database import connection as db_connection


@pytest.mark.asyncio
async def test_get_config(test_client, auth_token):
    """Test getting configuration."""
    cookies = {"auth_token": auth_token}
    response = test_client.get("/api/config", cookies=cookies)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_update_config(test_client, auth_token):
    """Test updating configuration."""
    cookies = {"auth_token": auth_token}
    config_data = {
        "LOG_LEVEL": "INFO"
    }

    print(f"[DEBUG] test_update_config engine id={id(db_connection.engine)} session_factory id={id(db_connection.AsyncSessionLocal)}")
    
    response = test_client.put("/api/config", json=config_data, cookies=cookies)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_get_config_unauthorized(test_client):
    """Test getting configuration without authentication."""
    response = test_client.get("/api/config")
    
    # Should redirect or return 401/403
    assert response.status_code in [302, 401, 403]


@pytest.mark.asyncio
async def test_get_ui_models(test_client, auth_token):
    """Test getting UI models."""
    cookies = {"auth_token": auth_token}
    response = test_client.get("/api/config/ui/models", cookies=cookies)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_reset_config(test_client, auth_token):
    """Test resetting configuration."""
    cookies = {"auth_token": auth_token}
    response = test_client.post("/api/config/reset", cookies=cookies)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

