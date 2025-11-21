"""
Tests for Claude Proxy Service
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from app.service.claude_proxy_service import (
    ClaudeProxyService,
    MessagesRequest,
    TokenCountRequest,
    Message,
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
    MessagesResponse,
    TokenCountResponse,
    Usage,
    SystemContent,
    Tool,
)
from app.config.config import settings
from app.exception.api_exceptions import ApiClientException


@pytest.fixture
def mock_key_manager():
    """Fixture for mock KeyManager."""
    from unittest.mock import MagicMock, AsyncMock

    mock = MagicMock()
    mock.get_key = AsyncMock(return_value="test_api_key")
    mock.handle_api_failure = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_fastapi_request(mock_key_manager):
    """Fixture for mock FastAPI Request with KeyManager in app.state."""
    from unittest.mock import MagicMock
    
    # Create mock without spec to allow arbitrary attributes
    request = MagicMock()
    # Ensure app and app.state are properly initialized as MagicMocks
    request.app = MagicMock()
    request.app.state = MagicMock()
    # Set the key_manager attribute directly - this ensures hasattr() works
    request.app.state.key_manager = mock_key_manager
    return request


@pytest.fixture
def mock_db_session():
    return AsyncMock()


@pytest.mark.asyncio
async def test_create_message_gemini_model(mock_fastapi_request, mock_db_session):
    """Test create_message with Gemini model."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        stream=False,
    )

    mock_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hi there!"}],
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    }

    with patch(
        "app.service.claude_proxy_service.GeminiApiClient"
    ) as MockGeminiClient, patch.object(
        service.gemini_response_handler, "handle_response"
    ) as mock_handle_response, patch(
        "app.service.claude_proxy_service.update_usage_stats", new_callable=AsyncMock
    ) as mock_update_usage:
        mock_client_instance = MagicMock()
        mock_client_instance.generate_content = AsyncMock(return_value=mock_response)
        MockGeminiClient.return_value = mock_client_instance
        
        # Mock the response handler to return OpenAI format
        mock_handle_response.return_value = {
            "choices": [{
                "message": {"content": "Hi there!", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            "id": "msg_test123"
        }

        response = await service.create_message(
            request_obj, mock_fastapi_request, session=mock_db_session
        )

        assert isinstance(response, MessagesResponse)
        assert response.content[0].text == "Hi there!"
        mock_client_instance.generate_content.assert_awaited_once()
        mock_update_usage.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_message_gemini_streaming(mock_fastapi_request, mock_db_session):
    """Test create_message with streaming for Gemini model."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        stream=True,
    )

    async def mock_stream():
        yield 'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}\n\n'
        yield "data: [DONE]\n\n"

    mock_client_instance = MagicMock()
    # stream_generate_content is an async generator function that returns an async generator
    # When called, it returns the async generator directly (not a coroutine)
    # Use MagicMock with return_value to return the async generator directly
    mock_client_instance.stream_generate_content = MagicMock(return_value=mock_stream())

    with patch(
        "app.service.claude_proxy_service.GeminiApiClient", return_value=mock_client_instance
    ) as MockGeminiClient, patch.object(
        service.gemini_response_handler, "handle_response"
    ) as mock_handle_response:
        # Mock the response handler to return OpenAI format for streaming
        mock_handle_response.return_value = {
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        }

        response = await service.create_message(
            request_obj, mock_fastapi_request, session=mock_db_session
        )
        
        assert isinstance(response, StreamingResponse)
        # Verify GeminiApiClient was instantiated
        MockGeminiClient.assert_called()
        
        # Verify stream_generate_content was called
        # Consume the streaming response to ensure stream_generate_content is called
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        # stream_generate_content is an async generator function, so it's called (not awaited)
        mock_client_instance.stream_generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_create_message_litellm_model(mock_fastapi_request, mock_db_session):
    """Test create_message with OpenAI/Anthropic model via LiteLLM."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model="openai/gpt-4",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        stream=False,
    )

    mock_litellm_response = {
        "choices": [
            {
                "message": {"content": "Hello from OpenAI!", "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
        "id": "chatcmpl-test123",
    }

    with patch("app.service.claude_proxy_service.litellm.acompletion") as mock_litellm:
        mock_litellm.return_value = mock_litellm_response

        response = await service.create_message(
            request_obj, mock_fastapi_request, session=mock_db_session
        )

        assert isinstance(response, MessagesResponse)
        assert response.content[0].text == "Hello from OpenAI!"
        mock_litellm.assert_awaited_once()


@pytest.mark.asyncio
async def test_count_tokens_gemini_model(mock_fastapi_request, mock_db_session):
    """Test count_tokens with Gemini model."""
    service = ClaudeProxyService()
    request_obj = TokenCountRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        messages=[Message(role="user", content="Hello, how are you?")],
    )

    mock_response = {"totalTokens": 8}

    with patch(
        "app.service.claude_proxy_service.GeminiApiClient"
    ) as MockGeminiClient, patch(
        "app.service.claude_proxy_service.update_usage_stats", new_callable=AsyncMock
    ) as mock_update_usage:
        mock_client_instance = MagicMock()
        mock_client_instance.count_tokens = AsyncMock(return_value=mock_response)
        MockGeminiClient.return_value = mock_client_instance

        response = await service.count_tokens(
            request_obj, mock_fastapi_request, session=mock_db_session
        )

        assert isinstance(response, TokenCountResponse)
        assert response.input_tokens == 8
        mock_client_instance.count_tokens.assert_awaited_once()


@pytest.mark.asyncio
async def test_count_tokens_litellm_model(mock_fastapi_request, mock_db_session):
    """Test count_tokens with OpenAI/Anthropic model via LiteLLM."""
    service = ClaudeProxyService()
    request_obj = TokenCountRequest(
        model="openai/gpt-4",
        messages=[Message(role="user", content="Hello")],
    )

    with patch(
        "app.service.claude_proxy_service.litellm.token_counter"
    ) as mock_token_counter:
        mock_token_counter.return_value = 10

        response = await service.count_tokens(
            request_obj, mock_fastapi_request, session=mock_db_session
        )

        assert isinstance(response, TokenCountResponse)
        assert response.input_tokens == 10
        mock_token_counter.assert_called_once()


@pytest.mark.asyncio
async def test_create_message_no_key_manager(mock_db_session):
    """Test create_message raises error when KeyManager is not available."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )

    # Use a simple object that doesn't have key_manager attribute
    # This simulates the real scenario where key_manager is not set on app.state
    class MockState:
        pass
    
    mock_request = MagicMock()
    mock_request.app = MagicMock()
    mock_request.app.state = MockState()
    # Don't set key_manager - this test verifies the error when it's missing

    with pytest.raises(RuntimeError, match="KeyManager not initialized"):
        await service.create_message(
            request_obj, mock_request, session=mock_db_session
        )


@pytest.mark.asyncio
async def test_create_message_no_api_key(mock_key_manager, mock_db_session):
    """Test create_message raises error when no API key is available."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )

    mock_key_manager.get_key = AsyncMock(return_value=None)
    mock_request = MagicMock()
    mock_request.app = MagicMock()
    mock_request.app.state = MagicMock()
    mock_request.app.state.key_manager = mock_key_manager

    with pytest.raises(HTTPException) as exc_info:
        await service.create_message(
            request_obj, mock_request, session=mock_db_session
        )

    assert exc_info.value.status_code == 429
    assert "exhausted" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_create_message_with_tools(mock_fastapi_request, mock_db_session):
    """Test create_message with tools."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[Message(role="user", content="What's the weather?")],
        tools=[
            Tool(
                name="get_weather",
                description="Get the weather",
                input_schema={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            )
        ],
        stream=False,
    )

    mock_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "I'll check the weather."}],
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }

    with patch(
        "app.service.claude_proxy_service.GeminiApiClient"
    ) as MockGeminiClient, patch(
        "app.service.claude_proxy_service.update_usage_stats", new_callable=AsyncMock
    ) as mock_update_usage:
        mock_client_instance = MagicMock()
        mock_client_instance.generate_content = AsyncMock(return_value=mock_response)
        MockGeminiClient.return_value = mock_client_instance

        response = await service.create_message(
            request_obj, mock_fastapi_request, session=mock_db_session
        )

        assert isinstance(response, MessagesResponse)
        # Verify tools were included in the payload
        call_args = mock_client_instance.generate_content.call_args
        assert call_args is not None
        payload = call_args[0][0]  # First positional argument is payload (generate_content(payload, model, api_key))
        assert "tools" in payload
        mock_update_usage.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_message_with_system(mock_fastapi_request, mock_db_session):
    """Test create_message with system instruction."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="You are a helpful assistant.",
        stream=False,
    )

    mock_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello! How can I help?"}],
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 6},
    }

    with patch(
        "app.service.claude_proxy_service.GeminiApiClient"
    ) as MockGeminiClient, patch(
        "app.service.claude_proxy_service.update_usage_stats", new_callable=AsyncMock
    ) as mock_update_usage:
        mock_client_instance = MagicMock()
        mock_client_instance.generate_content = AsyncMock(return_value=mock_response)
        MockGeminiClient.return_value = mock_client_instance

        response = await service.create_message(
            request_obj, mock_fastapi_request, session=mock_db_session
        )

        assert isinstance(response, MessagesResponse)
        # Verify system instruction was included in the payload
        call_args = mock_client_instance.generate_content.call_args
        assert call_args is not None
        payload = call_args[0][0]  # First positional argument is payload (generate_content(payload, model, api_key))
        assert "systemInstruction" in payload
        mock_update_usage.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_message_with_image(mock_fastapi_request, mock_db_session):
    """Test create_message with image content."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockText(type="text", text="What's in this image?"),
                    ContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                        },
                    ),
                ],
            )
        ],
        stream=False,
    )

    mock_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "This is a test image."}],
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 4},
    }

    with patch(
        "app.service.claude_proxy_service.GeminiApiClient"
    ) as MockGeminiClient, patch(
        "app.service.claude_proxy_service.update_usage_stats", new_callable=AsyncMock
    ) as mock_update_usage:
        mock_client_instance = MagicMock()
        mock_client_instance.generate_content = AsyncMock(return_value=mock_response)
        MockGeminiClient.return_value = mock_client_instance

        response = await service.create_message(
            request_obj, mock_fastapi_request, session=mock_db_session
        )

        assert isinstance(response, MessagesResponse)
        # Verify image was converted properly in payload
        call_args = mock_client_instance.generate_content.call_args
        assert call_args is not None
        payload = call_args[0][0]  # First positional argument is payload (generate_content(payload, model, api_key))
        contents = payload["contents"]
        assert len(contents) > 0
        # Check if inline_data is present (image converted to Gemini format)
        has_image = any(
            "inline_data" in part
            for content in contents
            for part in content.get("parts", [])
        )
        assert has_image
        mock_update_usage.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_message_with_tool_use(mock_fastapi_request, mock_db_session):
    """Test create_message with tool use in messages."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[
            Message(
                role="assistant",
                content=[
                    ContentBlockToolUse(
                        type="tool_use",
                        id="toolu_test123",
                        name="get_weather",
                        input={"location": "San Francisco"},
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    ContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="toolu_test123",
                        content="Sunny, 72°F",
                    )
                ],
            ),
        ],
        stream=False,
    )

    mock_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "It's sunny in San Francisco!"}],
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 7},
    }

    with patch(
        "app.service.claude_proxy_service.GeminiApiClient"
    ) as MockGeminiClient, patch(
        "app.service.claude_proxy_service.update_usage_stats", new_callable=AsyncMock
    ) as mock_update_usage:
        mock_client_instance = MagicMock()
        mock_client_instance.generate_content = AsyncMock(return_value=mock_response)
        MockGeminiClient.return_value = mock_client_instance

        response = await service.create_message(
            request_obj, mock_fastapi_request, session=mock_db_session
        )

        assert isinstance(response, MessagesResponse)
        # Verify tool use and tool result were converted properly
        call_args = mock_client_instance.generate_content.call_args
        assert call_args is not None
        payload = call_args[0][0]  # First positional argument is payload (generate_content(payload, model, api_key))
        contents = payload["contents"]
        # Should have function_call and function_response in parts
        has_function_call = any(
            "function_call" in part
            for content in contents
            for part in content.get("parts", [])
        )
        has_function_response = any(
            "function_response" in part
            for content in contents
            for part in content.get("parts", [])
        )
        assert has_function_call or has_function_response
        mock_update_usage.assert_awaited_once()
        mock_update_usage.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_message_error_handling(mock_fastapi_request, mock_db_session):
    """Test create_message error handling."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        stream=False,
    )

    with patch(
        "app.service.claude_proxy_service.GeminiApiClient"
    ) as MockGeminiClient, patch(
        "app.service.claude_proxy_service.add_error_log"
    ) as mock_add_error_log, patch(
        "app.service.claude_proxy_service.add_request_log"
    ) as mock_add_request_log:
        mock_client_instance = MagicMock()
        mock_client_instance.generate_content = AsyncMock(
            side_effect=ApiClientException(500, "Internal Server Error")
        )
        MockGeminiClient.return_value = mock_client_instance

        with pytest.raises(HTTPException) as exc_info:
            await service.create_message(
                request_obj, mock_fastapi_request, session=mock_db_session
            )

        assert exc_info.value.status_code == 500
        mock_add_error_log.assert_awaited_once()
        mock_add_request_log.assert_awaited_once()


@pytest.mark.asyncio
async def test_count_tokens_error_handling(mock_fastapi_request, mock_db_session):
    """Test count_tokens error handling."""
    service = ClaudeProxyService()
    request_obj = TokenCountRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        messages=[Message(role="user", content="Hello")],
    )

    with patch(
        "app.service.claude_proxy_service.GeminiApiClient"
    ) as MockGeminiClient, patch(
        "app.service.claude_proxy_service.add_error_log"
    ) as mock_add_error_log, patch(
        "app.service.claude_proxy_service.add_request_log"
    ) as mock_add_request_log:
        mock_client_instance = MagicMock()
        mock_client_instance.count_tokens = AsyncMock(
            side_effect=ApiClientException(500, "Internal Server Error")
        )
        MockGeminiClient.return_value = mock_client_instance

        with pytest.raises(HTTPException) as exc_info:
            await service.count_tokens(
                request_obj, mock_fastapi_request, session=mock_db_session
            )

        assert exc_info.value.status_code == 500
        mock_add_error_log.assert_awaited_once()
        mock_add_request_log.assert_awaited_once()


@pytest.mark.asyncio
async def test_count_tokens_unsupported_model(mock_fastapi_request, mock_db_session):
    """Test count_tokens with unsupported model prefix."""
    service = ClaudeProxyService()
    request_obj = TokenCountRequest(
        model="unsupported/model-name",
        messages=[Message(role="user", content="Hello")],
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.count_tokens(
            request_obj, mock_fastapi_request, session=mock_db_session
        )

    assert exc_info.value.status_code == 400
    assert "Unsupported model prefix" in exc_info.value.detail


def test_model_mapping_haiku():
    """Test that 'haiku' model names map to CLAUDE_SMALL_MODEL."""
    from app.service.claude_proxy_service import MessagesRequest

    request_obj = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )

    assert request_obj.model == settings.CLAUDE_SMALL_MODEL


def test_model_mapping_sonnet():
    """Test that 'sonnet' model names map to CLAUDE_BIG_MODEL."""
    from app.service.claude_proxy_service import MessagesRequest

    request_obj = MessagesRequest(
        model="claude-3-sonnet-20240229",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )

    assert request_obj.model == settings.CLAUDE_BIG_MODEL


def test_model_mapping_with_prefix():
    """Test that model names with prefix (gemini/, openai/, anthropic/) are preserved."""
    from app.service.claude_proxy_service import MessagesRequest

    request_obj = MessagesRequest(
        model="gemini/gemini-2.5-flash",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )

    assert request_obj.model == "gemini/gemini-2.5-flash"


@pytest.mark.asyncio
async def test_streaming_handler_tool_calls(mock_fastapi_request, mock_db_session):
    """Test streaming handler correctly processes tool calls."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[Message(role="user", content="Use the weather tool")],
        stream=True,
    )

    # Mock the streaming response from Gemini API
    async def mock_stream():
        yield 'data: {"candidates": [{"content": {"parts": [{"function_call": {"name": "get_weather", "args": {"location": "SF"}}}]}}]}\n\n'
        yield "data: [DONE]\n\n"

    mock_client_instance = MagicMock()
    # stream_generate_content is an async generator function that returns an async generator
    # When called, it returns the async generator directly (not a coroutine)
    # Use MagicMock with return_value to return the async generator directly
    mock_client_instance.stream_generate_content = MagicMock(return_value=mock_stream())

    with patch(
        "app.service.claude_proxy_service.GeminiApiClient", return_value=mock_client_instance
    ) as MockGeminiClient, patch.object(
        service.gemini_response_handler, "handle_response"
    ) as mock_handle_response, patch(
        "app.service.claude_proxy_service.litellm"
    ) as mock_litellm:
        # Mock the response handler to return OpenAI-format chunks
        mock_handle_response.return_value = {
            "choices": [
                {
                    "delta": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "toolu_test",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        response = await service.create_message(
            request_obj, mock_fastapi_request, session=mock_db_session
        )

        assert isinstance(response, StreamingResponse)
        # Verify GeminiApiClient was instantiated
        MockGeminiClient.assert_called()
        
        # Verify stream_generate_content was called
        # Consume the streaming response to ensure stream_generate_content is called
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        # stream_generate_content is an async generator function, so it's called (not awaited)
        mock_client_instance.stream_generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_from_gemini_to_anthropic():
    """Test conversion from Gemini response to Anthropic format."""
    service = ClaudeProxyService()
    
    gemini_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello from Gemini!"}],
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4},
    }
    
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )
    
    with patch(
        "app.service.claude_proxy_service.GeminiResponseHandler"
    ) as MockHandler:
        mock_handler_instance = MagicMock()
        mock_handler_instance.handle_response.return_value = {
            "choices": [
                {
                    "message": {"content": "Hello from Gemini!", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4},
            "id": "chatcmpl-test123",
        }
        service.gemini_response_handler = mock_handler_instance
        
        response = service._from_gemini_to_anthropic(gemini_response, request_obj)
        
        assert isinstance(response, MessagesResponse)
        assert response.content[0].text == "Hello from Gemini!"
        assert response.usage.input_tokens == 5
        assert response.usage.output_tokens == 4


def test_convert_anthropic_to_gemini_format():
    """Test conversion from Anthropic request to Gemini format."""
    service = ClaudeProxyService()
    request_obj = MessagesRequest(
        model=settings.CLAUDE_SMALL_MODEL,
        max_tokens=100,
        messages=[
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[ContentBlockText(type="text", text="Hi there!")],
            ),
        ],
        system="You are helpful",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
    )
    
    model_name, payload = service._convert_anthropic_to_gemini_format(request_obj)
    
    assert model_name == settings.CLAUDE_SMALL_MODEL
    assert "contents" in payload
    assert "generationConfig" in payload
    assert "systemInstruction" in payload
    assert payload["generationConfig"]["temperature"] == 0.7
    assert payload["generationConfig"]["topP"] == 0.9
    assert payload["generationConfig"]["topK"] == 40
    assert payload["generationConfig"]["maxOutputTokens"] == 100

