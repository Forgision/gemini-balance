from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException
from app.router import gemini_routes


@pytest.fixture(autouse=True)
def mock_auth_token(request):
    """
    Fixture to mock auth token verification across all tests in this module.
    Skips mocking if 'no_mock_auth' marker is present (for unauthorized tests).
    Note: gemini_routes uses security_service.verify_key_or_goog_api_key
    which is already mocked in conftest.py route_test_app fixture.
    """
    if "no_mock_auth" in request.keywords:
        yield
        return

    with (
        patch("app.core.security.verify_auth_token", return_value=True),
        patch("app.middleware.middleware.verify_auth_token", return_value=True),
    ):
        yield


# Test for the /models endpoint
def test_list_models_success(route_client, mocker):
    """Test successful retrieval of models."""
    mocker.patch(
        "app.service.model.model_service.ModelService.get_gemini_models",
        new_callable=AsyncMock,
        return_value={
            "models": [
                {
                    "name": "models/gemini-pro",
                    "displayName": "Gemini Pro",
                    "description": "The best model for scaling across a wide range of tasks.",
                }
            ]
        },
    )

    response = route_client.get("/gemini/v1beta/models")

    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0
    assert data["models"][0]["name"] == "models/gemini-pro"


def test_list_models_unauthorized(route_client):
    """Test unauthorized access to list_models."""
    # This test is a special case. It needs to test the security dependency failure itself.
    # The safest way to do this without interfering with other tests is to keep the local override,
    # but it's critical that the try/finally block is used to restore the original state.
    original_overrides = route_client.app.dependency_overrides.copy()
    try:

        async def override_security():
            raise HTTPException(status_code=401, detail="Unauthorized")

        route_client.app.dependency_overrides[
            gemini_routes.security_service.verify_key_or_goog_api_key
        ] = override_security

        response = route_client.get("/gemini/v1beta/models")
        assert response.status_code == 401
        assert response.json() == {
            "error": {"code": "http_error", "message": "Unauthorized"}
        }
    finally:
        # clean up
        route_client.app.dependency_overrides.clear()
        route_client.app.dependency_overrides.update(original_overrides)


def test_list_models_no_valid_key(route_client, route_mock_key_manager):
    """Test list_models when no valid API key is available."""
    # Configure the global mock to simulate the desired scenario for this test
    route_mock_key_manager.get_random_valid_key.return_value = None

    response = route_client.get("/gemini/v1beta/models")
    assert response.status_code == 503
    assert "No valid API keys available" in response.text


def test_list_models_derived_models(route_client):
    """Test that derived models are correctly added."""
    pass


# Tests for content generation
def test_generate_content_success(route_client, route_mock_chat_service):
    """Test successful content generation."""
    request_payload = {"contents": [{"parts": [{"text": "Hello"}]}]}
    response = route_client.post(
        "/gemini/v1beta/models/gemini-pro:generateContent", json=request_payload
    )

    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert data["candidates"][0]["content"]["parts"][0]["text"] == "Hello, world!"
    # Verify that the correct service method was called and session was passed
    route_mock_chat_service.generate_content.assert_called_once()
    # Note: We can't easily verify session was passed in route tests without mocker
    # The session verification is better suited for service-level tests


@pytest.mark.no_mock_auth
def test_generate_content_unauthorized(route_client):
    """Test unauthorized access to generate_content."""
    original_overrides = route_client.app.dependency_overrides.copy()
    try:

        async def override_security():
            raise HTTPException(status_code=401, detail="Unauthorized")

        route_client.app.dependency_overrides[
            gemini_routes.security_service.verify_key_or_goog_api_key
        ] = override_security

        request_payload = {"contents": [{"parts": [{"text": "Hello"}]}]}
        response = route_client.post(
            "/gemini/v1beta/models/gemini-pro:generateContent", json=request_payload
        )
        assert response.status_code == 401
    finally:
        route_client.app.dependency_overrides.clear()
        route_client.app.dependency_overrides.update(original_overrides)


def test_generate_content_unsupported_model(route_client, mocker):
    """Test content generation with an unsupported model."""
    # Patch the module-level model_service instance
    mock_check_model_support = mocker.patch(
        "app.router.gemini_routes.model_service.check_model_support",
        new_callable=AsyncMock,
        return_value=False,
    )

    request_payload = {"contents": [{"parts": [{"text": "Hello"}]}]}
    response = route_client.post(
        "/gemini/v1beta/models/unsupported-model:generateContent", json=request_payload
    )

    assert response.status_code == 400
    assert "Model unsupported-model is not supported" in response.text
    mock_check_model_support.assert_called_once_with("unsupported-model")


def test_stream_generate_content_success(route_client):
    """Test successful streaming content generation."""
    pass


# Tests for token counting
def test_count_tokens_success(route_client, route_mock_chat_service):
    """Test successful token counting."""
    request_payload = {"contents": [{"parts": [{"text": "Count these tokens"}]}]}
    response = route_client.post(
        "/gemini/v1beta/models/gemini-pro:countTokens", json=request_payload
    )

    assert response.status_code == 200
    data = response.json()
    assert "totalTokens" in data
    assert data["totalTokens"] == 123
    # Verify service was called
    route_mock_chat_service.count_tokens.assert_called_once()


@pytest.mark.no_mock_auth
def test_count_tokens_unauthorized(route_client):
    """Test unauthorized access to count_tokens."""
    original_overrides = route_client.app.dependency_overrides.copy()
    try:

        async def override_security():
            raise HTTPException(status_code=401, detail="Unauthorized")

        route_client.app.dependency_overrides[
            gemini_routes.security_service.verify_key_or_goog_api_key
        ] = override_security

        request_payload = {"contents": [{"parts": [{"text": "Count these tokens"}]}]}
        response = route_client.post(
            "/gemini/v1beta/models/gemini-pro:countTokens", json=request_payload
        )
        assert response.status_code == 401
    finally:
        route_client.app.dependency_overrides.clear()
        route_client.app.dependency_overrides.update(original_overrides)


# Tests for embedding
def test_embed_content_success(route_client, route_mock_embedding_service):
    """Test successful content embedding."""
    request_payload = {"content": {"parts": [{"text": "Embed this!"}]}}
    response = route_client.post(
        "/gemini/v1beta/models/gemini-pro:embedContent", json=request_payload
    )

    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert "values" in data["embedding"]
    assert len(data["embedding"]["values"]) == 3
    # Verify service was called
    route_mock_embedding_service.embed_content.assert_called_once()


@pytest.mark.no_mock_auth
def test_embed_content_unauthorized(route_client):
    """Test unauthorized access to embed_content."""
    original_overrides = route_client.app.dependency_overrides.copy()
    try:

        async def override_security():
            raise HTTPException(status_code=401, detail="Unauthorized")

        route_client.app.dependency_overrides[
            gemini_routes.security_service.verify_key_or_goog_api_key
        ] = override_security

        request_payload = {"content": {"parts": [{"text": "Embed this!"}]}}
        response = route_client.post(
            "/gemini/v1beta/models/gemini-pro:embedContent", json=request_payload
        )
        assert response.status_code == 401
    finally:
        route_client.app.dependency_overrides.clear()
        route_client.app.dependency_overrides.update(original_overrides)


def test_batch_embed_contents_success(route_client):
    """Test successful batch content embedding."""
    pass


# Tests for key management (These tests do not use the global key manager, they are for a different purpose)
def test_reset_all_key_fail_counts_success(route_client):
    """Test successful reset of all key fail counts."""
    original_overrides = route_client.app.dependency_overrides.copy()
    try:
        mock_key_manager = MagicMock()
        mock_key_manager.get_keys_by_status = AsyncMock(
            return_value={"valid_keys": {}, "invalid_keys": {}}
        )
        mock_key_manager.reset_key_failure_count = AsyncMock(return_value=True)

        async def override_get_key_manager():
            return mock_key_manager

        route_client.app.dependency_overrides[gemini_routes.get_key_manager] = (
            override_get_key_manager
        )

        response = route_client.post("/gemini/v1beta/reset-all-fail-counts")
        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "message": "Failure count for all keys has been reset.",
            "reset_count": 0,
        }
    finally:
        # clean up
        route_client.app.dependency_overrides.clear()
        route_client.app.dependency_overrides.update(original_overrides)


@pytest.mark.no_mock_auth
def test_reset_all_key_fail_counts_unauthorized(route_client):
    """Test unauthorized access to reset_all_key_fail_counts."""
    original_overrides = route_client.app.dependency_overrides.copy()
    try:

        async def override_security():
            raise HTTPException(status_code=401, detail="Unauthorized")

        route_client.app.dependency_overrides[
            gemini_routes.security_service.verify_key_or_goog_api_key
        ] = override_security

        response = route_client.post("/gemini/v1beta/reset-all-fail-counts")
        assert response.status_code == 401
    finally:
        route_client.app.dependency_overrides.clear()
        route_client.app.dependency_overrides.update(original_overrides)


def test_reset_selected_key_fail_counts_success(route_client):
    """Test successful reset of selected key fail counts."""
    pass


def test_reset_key_fail_count_success(route_client):
    """Test successful reset of a single key fail count."""
    pass


def test_verify_key_success(route_client):
    """Test successful key verification."""
    pass


def test_verify_selected_keys_success(route_client):
    """Test successful verification of selected keys."""
    pass
