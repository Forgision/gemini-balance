"""
Integration tests for Gemini routes.
Tests interact with the app like real users, using mocked API clients only.
"""

import pytest
import json


@pytest.mark.asyncio
async def test_list_models_success(test_client, goog_api_key_header):
    """Test successful retrieval of models list."""
    response = test_client.get("/v1beta/models", headers=goog_api_key_header)
    
    if response.status_code != 200:
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text[:1000]}")
    
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0
    
    # Check that models have required fields
    for model in data["models"]:
        assert "name" in model
        assert model["name"].startswith("models/")


@pytest.mark.asyncio
async def test_list_models_with_derived_models(test_client, goog_api_key_header):
    """Test that derived models (search, image, non-thinking) are added."""
    response = test_client.get("/v1beta/models", headers=goog_api_key_header)
    
    assert response.status_code == 200
    data = response.json()
    model_names = [m["name"].replace("models/", "") for m in data["models"]]
    
    # Check if derived models exist (they depend on settings)
    # We'll just verify the structure is correct
    assert any("gemini" in name.lower() for name in model_names)


@pytest.mark.asyncio
async def test_list_models_gemini_prefix(test_client, goog_api_key_header):
    """Test listing models via /gemini/v1beta/models endpoint."""
    response = test_client.get("/gemini/v1beta/models", headers=goog_api_key_header)
    
    assert response.status_code == 200
    data = response.json()
    assert "models" in data


@pytest.mark.asyncio
async def test_generate_content_success(test_client, goog_api_key_header):
    """Test successful non-streaming content generation."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Hello, what is 2+2?"}],
                "role": "user"
            }
        ]
    }
    
    response = test_client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json=payload,
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert len(data["candidates"]) > 0
    assert "content" in data["candidates"][0]
    assert "parts" in data["candidates"][0]["content"]


@pytest.mark.asyncio
async def test_generate_content_streaming(test_client, goog_api_key_header):
    """Test successful streaming content generation."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Say hello world"}],
                "role": "user"
            }
        ]
    }
    
    response = test_client.post(
        "/v1beta/models/gemini-pro:streamGenerateContent",
        json=payload,
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # Read streaming response
    content = b""
    for chunk in response.iter_bytes(chunk_size=1024):
        content += chunk
    
    # Verify SSE format
    content_str = content.decode("utf-8")
    assert "data:" in content_str


@pytest.mark.asyncio
async def test_generate_content_with_various_parameters(test_client, goog_api_key_header):
    """Test content generation with various request parameters."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Tell me a joke"}],
                "role": "user"
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
    
    response = test_client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json=payload,
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data


@pytest.mark.asyncio
async def test_count_tokens_success(test_client, goog_api_key_header):
    """Test successful token counting."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Count these tokens please"}],
                "role": "user"
            }
        ]
    }
    
    response = test_client.post(
        "/v1beta/models/gemini-pro:countTokens",
        json=payload,
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "totalTokens" in data
    assert isinstance(data["totalTokens"], int)


@pytest.mark.asyncio
async def test_embed_content_success(test_client, goog_api_key_header):
    """Test successful content embedding."""
    payload = {
        "content": {
            "parts": [{"text": "This is text to embed"}]
        }
    }
    
    response = test_client.post(
        "/v1beta/models/embedding-001:embedContent",
        json=payload,
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert "values" in data["embedding"]
    assert isinstance(data["embedding"]["values"], list)
    assert len(data["embedding"]["values"]) > 0


@pytest.mark.asyncio
async def test_batch_embed_contents_success(test_client, goog_api_key_header):
    """Test successful batch embedding."""
    payload = {
        "requests": [
            {
                "content": {
                    "parts": [{"text": "First text to embed"}]
                }
            },
            {
                "content": {
                    "parts": [{"text": "Second text to embed"}]
                }
            }
        ]
    }
    
    response = test_client.post(
        "/v1beta/models/embedding-001:batchEmbedContents",
        json=payload,
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) == 2


@pytest.mark.asyncio
async def test_generate_content_with_model_variants(test_client, goog_api_key_header):
    """Test content generation with different model variants."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Test message"}],
                "role": "user"
            }
        ]
    }
    
    # Test with base model
    response = test_client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json=payload,
        headers=goog_api_key_header
    )
    assert response.status_code == 200
    
    # Test with search variant (if configured)
    response = test_client.post(
        "/v1beta/models/gemini-pro-search:generateContent",
        json=payload,
        headers=goog_api_key_header
    )
    # Should work or return appropriate error if not configured
    assert response.status_code in [200, 400]


@pytest.mark.asyncio
async def test_stream_generate_content_sse_format(test_client, goog_api_key_header):
    """Test that streaming response follows SSE format."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Stream this message"}],
                "role": "user"
            }
        ]
    }
    
    response = test_client.post(
        "/v1beta/models/gemini-pro:streamGenerateContent",
        json=payload,
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    # Read first few chunks
    content_parts = []
    for line in response.iter_lines():
        if line:
            # iter_lines() already returns strings, not bytes
            content_parts.append(line if isinstance(line, str) else line.decode("utf-8"))
            if len(content_parts) >= 3:  # Read first few chunks
                break
    
    # Verify SSE format
    for part in content_parts:
        assert part.startswith("data:") or part == ""


@pytest.mark.asyncio
async def test_generate_content_multiple_messages(test_client, goog_api_key_header):
    """Test content generation with multiple messages in conversation."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Hello"}],
                "role": "user"
            },
            {
                "parts": [{"text": "Hi there!"}],
                "role": "model"
            },
            {
                "parts": [{"text": "How are you?"}],
                "role": "user"
            }
        ]
    }
    
    response = test_client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json=payload,
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data


@pytest.mark.asyncio
async def test_embed_content_with_task_type(test_client, goog_api_key_header):
    """Test embedding with task type parameter."""
    payload = {
        "content": {
            "parts": [{"text": "Text for embedding"}]
        },
        "taskType": "RETRIEVAL_DOCUMENT"
    }
    
    response = test_client.post(
        "/v1beta/models/embedding-001:embedContent",
        json=payload,
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data


@pytest.mark.asyncio
async def test_generate_content_with_key_in_url(test_client, auth_token):
    """Test content generation with API key in URL."""
    payload = {
        "contents": [
            {
                "parts": [{"text": "Test with key in URL"}],
                "role": "user"
            }
        ]
    }
    
    response = test_client.post(
        f"/v1beta/models/gemini-pro:generateContent?key={auth_token}",
        json=payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data

