from unittest.mock import AsyncMock, patch

def test_generate_content_success(client, mock_key_manager):
    """Test successful content generation."""
    request_body = {
        "contents": [{"parts": [{"text": "Hello"}]}],
    }
    mock_chat_service = AsyncMock()
    mock_chat_service.generate_content.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Hi there!"}]}}]
    }

    from app.dependencies import get_vertex_express_chat_service
    from app.router.vertex_express_routes import get_next_working_key
    app = client.app
    app.dependency_overrides[get_vertex_express_chat_service] = lambda: mock_chat_service
    app.dependency_overrides[get_next_working_key] = lambda: "test_api_key"


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
    del app.dependency_overrides[get_next_working_key]
