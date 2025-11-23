"""
Integration tests for OpenAI-compatible routes.
Tests interact with the app like real users, using mocked API clients only.
"""

import pytest


@pytest.mark.asyncio
async def test_list_models_openai_format(test_client, auth_header):
    """Test listing models in OpenAI format."""
    response = test_client.get("/v1/models", headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    # OpenAI format uses "data" key
    assert "data" in data or "object" in data


@pytest.mark.asyncio
async def test_list_models_hf_format(test_client, auth_header):
    """Test listing models via HuggingFace endpoint."""
    response = test_client.get("/hf/v1/models", headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    # OpenAI format uses "data" key
    assert "data" in data or "object" in data


@pytest.mark.asyncio
async def test_chat_completion_non_streaming(test_client, auth_header):
    """Test non-streaming chat completion."""
    payload = {
        "model": "gemini-pro",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stream": False
    }
    
    response = test_client.post("/v1/chat/completions", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]


@pytest.mark.asyncio
async def test_chat_completion_streaming(test_client, auth_header):
    """Test streaming chat completion."""
    payload = {
        "model": "gemini-pro",
        "messages": [
            {"role": "user", "content": "Say hello world"}
        ],
        "stream": True
    }
    
    response = test_client.post("/v1/chat/completions", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    # Read streaming response
    content_parts = []
    for line in response.iter_lines():
        if line:
            # iter_lines() already returns strings, not bytes
            content_parts.append(line if isinstance(line, str) else line.decode("utf-8"))
            if len(content_parts) >= 3:
                break
    
    # Verify SSE format
    for part in content_parts:
        assert part.startswith("data:") or part == "" or part == "[DONE]"


@pytest.mark.asyncio
async def test_chat_completion_with_parameters(test_client, auth_header):
    """Test chat completion with various parameters."""
    payload = {
        "model": "gemini-pro",
        "messages": [
            {"role": "user", "content": "Tell me a joke"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9,
        "stream": False
    }
    
    response = test_client.post("/v1/chat/completions", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data


@pytest.mark.asyncio
async def test_chat_completion_with_system_message(test_client, auth_header):
    """Test chat completion with system message."""
    payload = {
        "model": "gemini-pro",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "stream": False
    }
    
    response = test_client.post("/v1/chat/completions", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data


@pytest.mark.asyncio
async def test_chat_completion_multiple_turns(test_client, auth_header):
    """Test chat completion with multiple conversation turns."""
    payload = {
        "model": "gemini-pro",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ],
        "stream": False
    }
    
    response = test_client.post("/v1/chat/completions", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data


@pytest.mark.asyncio
async def test_chat_completion_hf_endpoint(test_client, auth_header):
    """Test chat completion via HuggingFace endpoint."""
    payload = {
        "model": "gemini-pro",
        "messages": [
            {"role": "user", "content": "Test message"}
        ],
        "stream": False
    }
    
    response = test_client.post("/hf/v1/chat/completions", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data


@pytest.mark.asyncio
async def test_create_embeddings(test_client, auth_header):
    """Test creating embeddings."""
    payload = {
        "model": "embedding-001",
        "input": "This is text to embed"
    }
    
    response = test_client.post("/v1/embeddings", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    assert "embedding" in data["data"][0]
    assert "usage" in data


@pytest.mark.asyncio
async def test_create_embeddings_multiple_inputs(test_client, auth_header):
    """Test creating embeddings with multiple inputs."""
    payload = {
        "model": "embedding-001",
        "input": [
            "First text to embed",
            "Second text to embed"
        ]
    }
    
    response = test_client.post("/v1/embeddings", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 2


@pytest.mark.asyncio
async def test_create_embeddings_hf_endpoint(test_client, auth_header):
    """Test embeddings via HuggingFace endpoint."""
    payload = {
        "model": "embedding-001",
        "input": "Text for embedding"
    }
    
    response = test_client.post("/hf/v1/embeddings", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data


@pytest.mark.asyncio
async def test_generate_image(test_client, auth_header):
    """Test image generation."""
    payload = {
        "prompt": "A beautiful sunset over mountains",
        "n": 1,
        "size": "1024x1024"
    }
    
    response = test_client.post("/v1/images/generations", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    # Image generation returns either "url" (if upload configured) or "b64_json" (if not)
    assert "url" in data["data"][0] or "b64_json" in data["data"][0]


@pytest.mark.asyncio
async def test_generate_image_hf_endpoint(test_client, auth_header):
    """Test image generation via HuggingFace endpoint."""
    payload = {
        "prompt": "A cat playing piano",
        "n": 1
    }
    
    response = test_client.post("/hf/v1/images/generations", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data


@pytest.mark.asyncio
async def test_list_keys_authenticated(test_client, auth_header):
    """Test listing keys (requires admin auth)."""
    response = test_client.get("/v1/keys/list", headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "data" in data


@pytest.mark.asyncio
async def test_list_keys_hf_endpoint(test_client, auth_header):
    """Test listing keys via HuggingFace endpoint."""
    response = test_client.get("/hf/v1/keys/list", headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


@pytest.mark.asyncio
async def test_chat_completion_image_chat(test_client, auth_header):
    """Test image chat completion."""
    payload = {
        "model": "gemini-2.0-flash-exp-chat",
        "messages": [
            {"role": "user", "content": "What's in this image?"}
        ],
        "stream": False
    }
    
    response = test_client.post("/v1/chat/completions", json=payload, headers=auth_header)
    
    # Should work or return error if model not configured
    assert response.status_code in [200, 400]

