"""
Integration tests for HuggingFace-compatible routes.
These are similar to OpenAI routes but with /hf/ prefix.
"""

import pytest


@pytest.mark.asyncio
async def test_hf_list_models(test_client, auth_header):
    """Test listing models via HuggingFace endpoint."""
    response = test_client.get("/hf/v1/models", headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data or "object" in data


@pytest.mark.asyncio
async def test_hf_chat_completion(test_client, auth_header):
    """Test chat completion via HuggingFace endpoint."""
    payload = {
        "model": "gemini-pro",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": False
    }
    
    response = test_client.post("/hf/v1/chat/completions", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data


@pytest.mark.asyncio
async def test_hf_chat_completion_streaming(test_client, auth_header):
    """Test streaming chat completion via HuggingFace endpoint."""
    payload = {
        "model": "gemini-pro",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "stream": True
    }
    
    response = test_client.post("/hf/v1/chat/completions", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_hf_embeddings(test_client, auth_header):
    """Test embeddings via HuggingFace endpoint."""
    payload = {
        "model": "embedding-001",
        "input": "Text to embed"
    }
    
    response = test_client.post("/hf/v1/embeddings", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data


@pytest.mark.asyncio
async def test_hf_image_generation(test_client, auth_header):
    """Test image generation via HuggingFace endpoint."""
    payload = {
        "prompt": "A beautiful landscape",
        "n": 1
    }
    
    response = test_client.post("/hf/v1/images/generations", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data


@pytest.mark.asyncio
async def test_hf_keys_list(test_client, auth_header):
    """Test listing keys via HuggingFace endpoint."""
    response = test_client.get("/hf/v1/keys/list", headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

