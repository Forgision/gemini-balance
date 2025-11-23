"""
Tests for Claude Proxy Routes
"""

from unittest.mock import AsyncMock, MagicMock

from app.config.config import settings
from app.service.claude_proxy_service import (
    ClaudeProxyService,
    ContentBlockText,
    MessagesResponse,
    TokenCountResponse,
)


def test_create_message_success(route_client):
    """Test successful message creation via Claude proxy."""
    request_body = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Mock the ClaudeProxyService
    mock_service = MagicMock(spec=ClaudeProxyService)
    mock_response = MessagesResponse(
        id="msg_test123",
        model="claude-3-haiku-20240307",
        content=[ContentBlockText(type="text", text="Hi there!")],
        stop_reason="end_turn",
        usage={"input_tokens": 5, "output_tokens": 3},
    )
    mock_service.create_message = AsyncMock(return_value=mock_response)

    from app.service.claude_proxy_service import ClaudeProxyService as ServiceClass

    app = route_client.app
    app.dependency_overrides[ServiceClass] = lambda: mock_service

    response = route_client.post(
        "/claude/v1/messages",
        json=request_body,
        headers={"Authorization": f"Bearer {settings.AUTH_TOKEN}"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["id"] == "msg_test123"
    assert result["content"][0]["text"] == "Hi there!"
    mock_service.create_message.assert_awaited_once()
    create_kwargs = mock_service.create_message.await_args.kwargs
    assert "session" in create_kwargs

    # Clean up
    del app.dependency_overrides[ServiceClass]


def test_create_message_streaming(route_client):
    """Test streaming message creation via Claude proxy."""
    request_body = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    # Mock streaming response
    async def mock_stream():
        yield 'event: message_start\ndata: {"type": "message_start"}\n\n'
        yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}\n\n'
        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    mock_service = MagicMock(spec=ClaudeProxyService)
    from fastapi.responses import StreamingResponse

    mock_service.create_message = AsyncMock(
        return_value=StreamingResponse(mock_stream(), media_type="text/event-stream")
    )

    from app.service.claude_proxy_service import ClaudeProxyService as ServiceClass

    app = route_client.app
    app.dependency_overrides[ServiceClass] = lambda: mock_service

    response = route_client.post(
        "/claude/v1/messages",
        json=request_body,
        headers={"Authorization": f"Bearer {settings.AUTH_TOKEN}"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    mock_service.create_message.assert_awaited_once()

    # Clean up
    del app.dependency_overrides[ServiceClass]


def test_count_tokens_success(route_client):
    """Test successful token counting via Claude proxy."""
    request_body = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }

    mock_service = MagicMock(spec=ClaudeProxyService)
    mock_response = TokenCountResponse(input_tokens=8)
    mock_service.count_tokens = AsyncMock(return_value=mock_response)

    from app.service.claude_proxy_service import ClaudeProxyService as ServiceClass

    app = route_client.app
    app.dependency_overrides[ServiceClass] = lambda: mock_service

    response = route_client.post(
        "/claude/v1/messages/count_tokens",
        json=request_body,
        headers={"Authorization": f"Bearer {settings.AUTH_TOKEN}"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["input_tokens"] == 8
    mock_service.count_tokens.assert_awaited_once()
    count_kwargs = mock_service.count_tokens.await_args.kwargs
    assert "session" in count_kwargs

    # Clean up
    del app.dependency_overrides[ServiceClass]


def test_create_message_with_tools(route_client):
    """Test message creation with tools."""
    request_body = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ],
        "stream": False,
    }

    mock_service = MagicMock(spec=ClaudeProxyService)
    mock_response = MessagesResponse(
        id="msg_test456",
        model="claude-3-haiku-20240307",
        content=[ContentBlockText(type="text", text="I'll check the weather.")],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    mock_service.create_message = AsyncMock(return_value=mock_response)

    from app.service.claude_proxy_service import ClaudeProxyService as ServiceClass

    app = route_client.app
    app.dependency_overrides[ServiceClass] = lambda: mock_service

    response = route_client.post(
        "/claude/v1/messages",
        json=request_body,
        headers={"Authorization": f"Bearer {settings.AUTH_TOKEN}"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["content"][0]["text"] == "I'll check the weather."
    mock_service.create_message.assert_awaited_once()

    # Clean up
    del app.dependency_overrides[ServiceClass]


def test_create_message_with_system(route_client):
    """Test message creation with system instruction."""
    request_body = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        "system": "You are a helpful assistant.",
        "stream": False,
    }

    mock_service = MagicMock(spec=ClaudeProxyService)
    mock_response = MessagesResponse(
        id="msg_test789",
        model="claude-3-haiku-20240307",
        content=[ContentBlockText(type="text", text="Hello! How can I help?")],
        stop_reason="end_turn",
        usage={"input_tokens": 12, "output_tokens": 6},
    )
    mock_service.create_message = AsyncMock(return_value=mock_response)

    from app.service.claude_proxy_service import ClaudeProxyService as ServiceClass

    app = route_client.app
    app.dependency_overrides[ServiceClass] = lambda: mock_service

    response = route_client.post(
        "/claude/v1/messages",
        json=request_body,
        headers={"Authorization": f"Bearer {settings.AUTH_TOKEN}"},
    )

    assert response.status_code == 200
    result = response.json()
    assert "Hello! How can I help?" in result["content"][0]["text"]
    mock_service.create_message.assert_awaited_once()

    # Clean up
    del app.dependency_overrides[ServiceClass]


def test_create_message_unauthorized(route_client):
    """Test message creation without authentication."""
    request_body = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = route_client.post(
        "/claude/v1/messages",
        json=request_body,
    )

    assert response.status_code == 401


def test_count_tokens_unauthorized(route_client):
    """Test token counting without authentication."""
    request_body = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = route_client.post(
        "/claude/v1/messages/count_tokens",
        json=request_body,
    )

    assert response.status_code == 401


def test_create_message_invalid_model(route_client):
    """Test message creation with invalid model."""
    request_body = {
        "model": "invalid-model",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    mock_service = MagicMock(spec=ClaudeProxyService)
    from fastapi import HTTPException

    mock_service.create_message = AsyncMock(
        side_effect=HTTPException(status_code=400, detail="Invalid model")
    )

    from app.service.claude_proxy_service import ClaudeProxyService as ServiceClass

    app = route_client.app
    app.dependency_overrides[ServiceClass] = lambda: mock_service

    response = route_client.post(
        "/claude/v1/messages",
        json=request_body,
        headers={"Authorization": f"Bearer {settings.AUTH_TOKEN}"},
    )

    assert response.status_code == 400
    mock_service.create_message.assert_awaited_once()

    # Clean up
    del app.dependency_overrides[ServiceClass]


def test_count_tokens_invalid_request(route_client):
    """Test token counting with invalid request."""
    request_body = {
        "model": "claude-3-haiku-20240307",
        # Missing required 'messages' field
    }

    response = route_client.post(
        "/claude/v1/messages/count_tokens",
        json=request_body,
        headers={"Authorization": f"Bearer {settings.AUTH_TOKEN}"},
    )

    assert response.status_code == 422  # Validation error


def test_create_message_model_mapping(route_client):
    """Test that model names are correctly mapped (haiku -> CLAUDE_SMALL_MODEL, sonnet -> CLAUDE_BIG_MODEL)."""
    request_body = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    mock_service = MagicMock(spec=ClaudeProxyService)
    mock_response = MessagesResponse(
        id="msg_test_mapping",
        model=settings.CLAUDE_SMALL_MODEL,
        content=[ContentBlockText(type="text", text="Hello!")],
        stop_reason="end_turn",
        usage={"input_tokens": 5, "output_tokens": 2},
    )
    mock_service.create_message = AsyncMock(return_value=mock_response)

    from app.service.claude_proxy_service import ClaudeProxyService as ServiceClass

    app = route_client.app
    app.dependency_overrides[ServiceClass] = lambda: mock_service

    response = route_client.post(
        "/claude/v1/messages",
        json=request_body,
        headers={"Authorization": f"Bearer {settings.AUTH_TOKEN}"},
    )

    assert response.status_code == 200
    # Verify the service was called
    mock_service.create_message.assert_awaited_once()
    call_args = mock_service.create_message.call_args
    assert call_args is not None
    request_obj = call_args[0][0]
    # The model should be mapped to the actual Gemini model name
    assert request_obj.model == settings.CLAUDE_SMALL_MODEL

    # Clean up
    del app.dependency_overrides[ServiceClass]
