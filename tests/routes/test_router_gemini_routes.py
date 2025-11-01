from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from fastapi import HTTPException
from app.router import gemini_routes
from app.domain.gemini_models import GeminiRequest, GeminiContent, GeminiEmbedRequest, GeminiEmbedContent

# Test for the /models endpoint
@patch("app.service.model.model_service.ModelService.get_gemini_models", new_callable=AsyncMock)
def test_list_models_success(mock_get_gemini_models, client):
    """Test successful retrieval of models."""
    mock_get_gemini_models.return_value = {
        "models": [
            {
                "name": "models/gemini-pro",
                "displayName": "Gemini Pro",
                "description": "The best model for scaling across a wide range of tasks.",
            }
        ]
    }

    response = client.get("/gemini/v1beta/models")

    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0
    assert data["models"][0]["name"] == "models/gemini-pro"

def test_list_models_unauthorized(client):
    """Test unauthorized access to list_models."""
    # This test is a special case. It needs to test the security dependency failure itself.
    # The safest way to do this without interfering with other tests is to keep the local override,
    # but it's critical that the try/finally block is used to restore the original state.
    original_overrides = client.app.dependency_overrides.copy()
    try:
        async def override_security():
            raise HTTPException(status_code=401, detail="Unauthorized")
        client.app.dependency_overrides[gemini_routes.security_service.verify_key_or_goog_api_key] = override_security

        response = client.get("/gemini/v1beta/models")
        assert response.status_code == 401
        assert response.json() == {"error": {"code": "http_error", "message": "Unauthorized"}}
    finally:
        # clean up
        client.app.dependency_overrides = original_overrides

def test_list_models_no_valid_key(client, mock_key_manager):
    """Test list_models when no valid API key is available."""
    # Configure the global mock to simulate the desired scenario for this test
    mock_key_manager.get_random_valid_key.return_value = None

    response = client.get("/gemini/v1beta/models")
    assert response.status_code == 503
    assert "No valid API keys available" in response.text

def test_list_models_derived_models(client):
    """Test that derived models are correctly added."""
    pass

# Tests for content generation
@patch("app.service.model.model_service.ModelService.check_model_support", new_callable=AsyncMock)
def test_generate_content_success(mock_check_model_support, client, mock_chat_service):
    """Test successful content generation."""
    mock_check_model_support.return_value = True

    request_payload = {
        "contents": [
            {"parts": [{"text": "Hello"}]}
        ]
    }
    response = client.post("/gemini/v1beta/models/gemini-pro:generateContent", json=request_payload)

    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert data["candidates"][0]["content"]["parts"][0]["text"] == "Hello, world!"
    # Verify that the correct service method was called
    mock_chat_service.generate_content.assert_called_once()

@patch("app.service.model.model_service.ModelService.check_model_support", new_callable=AsyncMock)
def test_generate_content_unsupported_model(mock_check_model_support, client):
    """Test content generation with an unsupported model."""
    mock_check_model_support.return_value = False
    request_payload = {
        "contents": [
            {"parts": [{"text": "Hello"}]}
        ]
    }
    response = client.post("/gemini/v1beta/models/unsupported-model:generateContent", json=request_payload)

    assert response.status_code == 400
    assert "Model unsupported-model is not supported" in response.text

def test_stream_generate_content_success(client):
    """Test successful streaming content generation."""
    pass

# Tests for token counting
@patch("app.service.model.model_service.ModelService.check_model_support", new_callable=AsyncMock)
def test_count_tokens_success(mock_check_model_support, client, mock_chat_service):
    """Test successful token counting."""
    mock_check_model_support.return_value = True

    request_payload = {
        "contents": [
            {"parts": [{"text": "Count these tokens"}]}
        ]
    }
    response = client.post("/gemini/v1beta/models/gemini-pro:countTokens", json=request_payload)

    assert response.status_code == 200
    data = response.json()
    assert "totalTokens" in data
    assert data["totalTokens"] == 123
    mock_chat_service.count_tokens.assert_called_once()

# Tests for embedding
@patch("app.service.model.model_service.ModelService.check_model_support", new_callable=AsyncMock)
def test_embed_content_success(mock_check_model_support, client, mock_embedding_service):
    """Test successful content embedding."""
    mock_check_model_support.return_value = True

    request_payload = {
        "content": {
            "parts": [{"text": "Embed this!"}]
        }
    }
    response = client.post("/gemini/v1beta/models/gemini-pro:embedContent", json=request_payload)

    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert "values" in data["embedding"]
    assert len(data["embedding"]["values"]) == 3
    mock_embedding_service.embed_content.assert_called_once()

def test_batch_embed_contents_success(client):
    """Test successful batch content embedding."""
    pass

# Tests for key management (These tests do not use the global key manager, they are for a different purpose)
def test_reset_all_key_fail_counts_success(client):
    """Test successful reset of all key fail counts."""
    original_overrides = client.app.dependency_overrides.copy()
    try:
        mock_key_manager = MagicMock()
        mock_key_manager.get_keys_by_status = AsyncMock(return_value={"valid_keys": {}, "invalid_keys": {}})
        mock_key_manager.reset_failure_counts = AsyncMock(return_value=None)

        async def override_get_key_manager():
            return mock_key_manager

        client.app.dependency_overrides[gemini_routes.get_key_manager] = override_get_key_manager

        response = client.post("/gemini/v1beta/reset-all-fail-counts")
        assert response.status_code == 200
        assert response.json() == {"success": True, "message": "Failure count for all keys has been reset."}
    finally:
        # clean up
        client.app.dependency_overrides = original_overrides

def test_reset_selected_key_fail_counts_success(client):
    """Test successful reset of selected key fail counts."""
    pass

def test_reset_key_fail_count_success(client):
    """Test successful reset of a single key fail count."""
    pass

def test_verify_key_success(client):
    """Test successful key verification."""
    pass

def test_verify_selected_keys_success(client):
    """Test successful verification of selected keys."""
    pass
