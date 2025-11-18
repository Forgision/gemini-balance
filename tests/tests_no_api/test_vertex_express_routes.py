"""
Integration tests for Vertex Express routes.
"""

import pytest


@pytest.mark.asyncio
async def test_vertex_express_list_models(test_client, goog_api_key_header):
    """Test listing models via Vertex Express endpoint."""
    response = test_client.get("/vertex-express/v1beta/models", headers=goog_api_key_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "models" in data


@pytest.mark.asyncio
async def test_vertex_express_generate_content(test_client, goog_api_key_header):
    """Test content generation via Vertex Express endpoint."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Hello"}],
                "role": "user"
            }
        ]
    }
    
    response = test_client.post(
        "/vertex-express/v1beta/models/gemini-pro:generateContent",
        json=payload,
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data


@pytest.mark.asyncio
async def test_vertex_express_stream_generate_content(test_client, goog_api_key_header):
    """Test streaming content generation via Vertex Express endpoint."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Say hello"}],
                "role": "user"
            }
        ]
    }
    
    response = test_client.post(
        "/vertex-express/v1beta/models/gemini-pro:streamGenerateContent",
        json=payload,
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

