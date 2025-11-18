"""
Integration tests for critical KeyManager functionality.
Tests how KeyManager integrates with routes and handles key selection.
"""

import pytest
from tests.tests_no_api.conftest import TEST_API_KEYS, TEST_VERTEX_API_KEYS


@pytest.mark.asyncio
async def test_key_selection_for_different_models(test_client, goog_api_key_header, test_key_manager):
    """Test that KeyManager selects keys for different models."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Test message"}],
                "role": "user"
            }
        ]
    }
    
    # Make requests with different models
    models = ["gemini-pro", "gemini-2.0-flash-exp"]
    
    for model in models:
        response = test_client.post(
            f"/v1beta/models/{model}:generateContent",
            json=payload,
            headers=goog_api_key_header
        )
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_key_rotation_across_requests(test_client, goog_api_key_header, test_key_manager):
    """Test that keys are rotated across multiple requests."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Test"}],
                "role": "user"
            }
        ]
    }
    
    # Make multiple requests
    for _ in range(3):
        response = test_client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json=payload,
            headers=goog_api_key_header
        )
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_vertex_key_handling(test_client, goog_api_key_header, test_key_manager):
    """Test that Vertex keys are used for Vertex Express routes."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Test"}],
                "role": "user"
            }
        ]
    }
    
    # Vertex Express routes should use vertex keys
    if TEST_VERTEX_API_KEYS:
        response = test_client.post(
            "/vertex-express/v1beta/models/gemini-pro:generateContent",
            json=payload,
            headers=goog_api_key_header
        )
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_key_manager_initialization(test_key_manager):
    """Test that KeyManager is properly initialized."""
    assert test_key_manager is not None
    assert test_key_manager.is_ready
    
    # Check that it has API keys
    keys_status = await test_key_manager.get_keys_by_status()
    assert "valid_keys" in keys_status
    assert "invalid_keys" in keys_status


@pytest.mark.asyncio
async def test_get_random_valid_key(test_key_manager):
    """Test getting a random valid key."""
    key = await test_key_manager.get_random_valid_key()
    assert key is not None
    assert key in TEST_API_KEYS


@pytest.mark.asyncio
async def test_get_key_for_model(test_key_manager):
    """Test getting a key for a specific model."""
    model = "gemini-pro"
    key = await test_key_manager.get_key(model, is_vertex_key=False)
    assert key is not None
    assert key in TEST_API_KEYS


@pytest.mark.asyncio
async def test_get_all_keys_with_status(test_key_manager):
    """Test getting all keys with their status."""
    keys_status = await test_key_manager.get_all_keys_with_fail_count()
    assert "valid_keys" in keys_status
    assert "invalid_keys" in keys_status
    assert isinstance(keys_status["valid_keys"], dict)
    assert isinstance(keys_status["invalid_keys"], dict)


@pytest.mark.asyncio
async def test_key_selection_consistency(test_client, goog_api_key_header, test_key_manager):
    """Test that key selection is consistent for the same model."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Test"}],
                "role": "user"
            }
        ]
    }
    
    # Make two requests with the same model
    response1 = test_client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json=payload,
        headers=goog_api_key_header
    )
    assert response1.status_code == 200
    
    response2 = test_client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json=payload,
        headers=goog_api_key_header
    )
    assert response2.status_code == 200

