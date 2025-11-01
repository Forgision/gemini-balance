from unittest.mock import AsyncMock
import pytest
from fastapi import HTTPException

# Test for the /openai/v1/models endpoint
def test_list_models_success(client, mock_key_manager, mocker):
    """Test successful retrieval of models."""
    mock_models_response = {
        "object": "list",
        "data": [
            {"id": "gemini-pro", "object": "model", "created": 1677610602, "owned_by": "google"},
            {"id": "gemini-pro-vision", "object": "model", "created": 1677610602, "owned_by": "google"},
        ],
    }
    mock_get_models = mocker.patch(
        "app.service.openai_compatiable.openai_compatiable_service.OpenAICompatiableService.get_models",
        new_callable=AsyncMock,
        return_value=mock_models_response
    )

    response = client.get("/openai/v1/models")

    assert response.status_code == 200
    assert response.json() == mock_models_response
    mock_key_manager.get_random_valid_key.assert_awaited_once()
    mock_get_models.assert_awaited_once_with("test_api_key")

def test_list_models_unauthorized(client, test_app):
    """Test unauthorized access to list_models."""
    from app.router import openai_compatible_routes

    async def override_auth_fail():
        raise HTTPException(status_code=401, detail="Unauthorized")

    original_override = test_app.dependency_overrides.get(openai_compatible_routes.security_service.verify_authorization)
    test_app.dependency_overrides[openai_compatible_routes.security_service.verify_authorization] = override_auth_fail

    response = client.get("/openai/v1/models")
    assert response.status_code == 401

    # Clean up the override
    if original_override:
        test_app.dependency_overrides[openai_compatible_routes.security_service.verify_authorization] = original_override
    else:
        del test_app.dependency_overrides[openai_compatible_routes.security_service.verify_authorization]


# Tests for chat completion
def test_chat_completion_success(client, mock_key_manager, mocker):
    """Test successful chat completion."""
    mock_chat_response = {
        "id": "chatcmpl-123", "object": "chat.completion", "created": 1677652288, "model": "gemini-pro",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello! How can I assist you today?"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    mock_create_chat = mocker.patch(
        "app.service.openai_compatiable.openai_compatiable_service.OpenAICompatiableService.create_chat_completion",
        new_callable=AsyncMock,
        return_value=mock_chat_response
    )
    chat_request_payload = {"model": "gemini-pro", "messages": [{"role": "user", "content": "Hello!"}]}

    response = client.post("/openai/v1/chat/completions", json=chat_request_payload)

    assert response.status_code == 200
    assert response.json() == mock_chat_response
    mock_key_manager.get_next_working_key.assert_awaited_once_with(model_name="gemini-pro")
    mock_create_chat.assert_awaited_once()

def test_chat_completion_image_chat_success(client, mock_key_manager, mocker):
    """Test successful image chat completion."""
    mock_image_chat_response = {
        "id": "chatcmpl-456", "object": "chat.completion", "created": 1677652288, "model": "imagen-3.0-generate-002-chat",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Here is the image you requested."}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }
    mock_create_image_chat = mocker.patch(
        "app.service.openai_compatiable.openai_compatiable_service.OpenAICompatiableService.create_image_chat_completion",
        new_callable=AsyncMock,
        return_value=mock_image_chat_response
    )
    chat_request_payload = {
        "model": "imagen-3.0-generate-002-chat",
        "messages": [{"role": "user", "content": "Generate an image of a cat."}],
    }

    response = client.post("/openai/v1/chat/completions", json=chat_request_payload)

    assert response.status_code == 200
    assert response.json() == mock_image_chat_response
    mock_key_manager.get_paid_key.assert_awaited_once()
    mock_create_image_chat.assert_awaited_once()

def test_chat_completion_stream_success(client, mock_key_manager, mocker):
    """Test successful streaming chat completion."""
    async def mock_stream_generator():
        yield "data: chunk 1"
        yield "data: chunk 2"

    mock_create_chat = mocker.patch(
        "app.service.openai_compatiable.openai_compatiable_service.OpenAICompatiableService.create_chat_completion",
        new_callable=AsyncMock,
        return_value=mock_stream_generator()
    )
    chat_request_payload = {
        "model": "gemini-pro",
        "messages": [{"role": "user", "content": "Tell me a story."}],
        "stream": True,
    }

    response = client.post("/openai/v1/chat/completions", json=chat_request_payload)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    streamed_content = response.text
    expected_content = "data: chunk 1data: chunk 2"
    assert streamed_content == expected_content
    mock_key_manager.get_next_working_key.assert_awaited_once_with(model_name="gemini-pro")
    mock_create_chat.assert_awaited_once()

# Tests for image generation
def test_generate_image_success(client, mock_key_manager, mocker):
    """Test successful image generation."""
    mock_image_response = {"created": 1677652288, "data": [{"url": "http://example.com/image.png"}]}
    mock_generate_images = mocker.patch(
        "app.service.openai_compatiable.openai_compatiable_service.OpenAICompatiableService.generate_images",
        new_callable=AsyncMock,
        return_value=mock_image_response
    )
    image_request_payload = {"prompt": "A picture of a cat.", "n": 1, "size": "1024x1024"}

    response = client.post("/openai/v1/images/generations", json=image_request_payload)

    assert response.status_code == 200
    assert response.json() == mock_image_response
    mock_key_manager.get_paid_key.assert_awaited_once()
    mock_generate_images.assert_awaited_once()

# Tests for embedding
def test_embedding_success(client, mock_key_manager, mocker):
    """Test successful text embedding."""
    mock_embedding_response = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }
    mock_create_embeddings = mocker.patch(
        "app.service.openai_compatiable.openai_compatiable_service.OpenAICompatiableService.create_embeddings",
        new_callable=AsyncMock,
        return_value=mock_embedding_response
    )
    embedding_request_payload = {
        "input": "The quick brown fox jumps over the lazy dog",
        "model": "text-embedding-ada-002",
    }

    response = client.post("/openai/v1/embeddings", json=embedding_request_payload)

    assert response.status_code == 200
    assert response.json() == mock_embedding_response
    mock_key_manager.get_next_working_key.assert_awaited_once_with(model_name="text-embedding-ada-002")
    mock_create_embeddings.assert_awaited_once()
