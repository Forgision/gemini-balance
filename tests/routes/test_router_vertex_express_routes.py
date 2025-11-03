import pytest
from unittest.mock import AsyncMock, patch

from app.router.vertex_express_routes import security_service

def test_generate_content_success(client):
    """Test successful content generation."""
    from app.domain.gemini_models import GeminiRequest, GeminiContent

    app = client.app
    app.dependency_overrides[
        security_service.verify_key_or_goog_api_key
    ] = lambda: "test_token"

    request_body = {
        "model": "gemini-pro",
        "request": GeminiRequest(
            contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])]
        ).model_dump(),
    }
    mock_chat_service = AsyncMock()
    mock_chat_service.generate_content.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Hi there!"}]}}]
    }

    from app.dependencies import get_vertex_express_chat_service
    from app.router.vertex_express_routes import dep_get_next_working_vertex_key

    app = client.app
    app.dependency_overrides[
        get_vertex_express_chat_service
    ] = lambda: mock_chat_service
    app.dependency_overrides[dep_get_next_working_vertex_key] = lambda: "test_api_key"

    response = client.post(
        "/vertex-express/v1beta/models/gemini-pro:generateContent",
        json=request_body,
        headers={"x-goog-api-key": "test_token"},
    )

    assert response.status_code == 200
    assert response.json()["candidates"][0]["content"]["parts"][0]["text"] == "Hi there!"
    mock_chat_service.generate_content.assert_awaited_once()

    # Clean up the override
    del app.dependency_overrides[get_vertex_express_chat_service]
    del app.dependency_overrides[dep_get_next_working_vertex_key]