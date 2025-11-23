from unittest.mock import AsyncMock, patch

from app.config.config import settings


def test_list_models_success(
    mock_verify_auth_token, route_client, route_mock_key_manager
):
    """Test the /models endpoint returns successfully."""
    with patch(
        "app.router.openai_routes.model_service.get_gemini_openai_models",
        new_callable=AsyncMock,
    ) as mock_get_models:
        mock_get_models.return_value = {"data": [{"id": "gpt-3.5-turbo"}]}
        response = route_client.get(
            "/hf/v1/models",
            headers={"Authorization": f"Bearer {settings.AUTH_TOKEN}"},
        )
        assert response.status_code == 200
        assert response.json()["data"][0]["id"] == "gpt-3.5-turbo"
        route_mock_key_manager.get_random_valid_key.assert_awaited_once()
        mock_get_models.assert_awaited_once_with("test_api_key")


def test_chat_completions_success(
    mock_verify_auth_token, route_client, route_mock_key_manager
):
    """Test successful chat completion."""
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    mock_chat_service = AsyncMock()
    mock_chat_service.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Hi there!"}}]
    }

    from app.dependencies import get_openai_chat_service
    from app.router.openai_routes import get_next_working_key_wrapper

    app = route_client.app
    app.dependency_overrides[get_openai_chat_service] = lambda: mock_chat_service
    app.dependency_overrides[get_next_working_key_wrapper] = lambda: "test_api_key"

    response = route_client.post(
        "/hf/v1/chat/completions",
        json=request_body,
        headers={"Authorization": f"Bearer {settings.AUTH_TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hi there!"
    mock_chat_service.create_chat_completion.assert_awaited_once()

    # Clean up the override
    del app.dependency_overrides[get_openai_chat_service]
    del app.dependency_overrides[get_next_working_key_wrapper]
