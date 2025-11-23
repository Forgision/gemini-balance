import pytest
from unittest.mock import AsyncMock, patch

from app.service.chat.vertex_express_chat_service import (
    GeminiChatService,
    _build_payload,
    _build_tools,
    _clean_json_schema_properties,
    _get_real_model,
    _get_safety_settings,
    _has_image_parts,
)
from app.domain.gemini_models import GeminiContent, GeminiRequest
from app.core.constants import GEMINI_2_FLASH_EXP_SAFETY_SETTINGS
from app.config.config import settings


@pytest.mark.asyncio
async def test_vertex_express_chat_service_generate_content(mock_key_manager):
    """Test the GeminiChatService (from vertex file) generate_content method."""
    with patch("app.service.chat.vertex_express_chat_service.GeminiApiClient.generate_content", new_callable=AsyncMock) as mock_generate_content:
        mock_generate_content.return_value = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
        response = await service.generate_content("gemini-pro", request, "test_api_key")
        mock_generate_content.assert_called_once()
        assert response["candidates"][0]["content"]["parts"][0]["text"] == "Hello"


@pytest.mark.asyncio
async def test_vertex_express_chat_service_stream_content(mock_key_manager):
    """Test the GeminiChatService (from vertex file) stream_generate_content method."""
    async def mock_stream():
        yield 'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}'

    with patch("app.service.chat.vertex_express_chat_service.GeminiApiClient.stream_generate_content") as mock_stream_generate_content:
        mock_stream_generate_content.return_value = mock_stream()
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
        stream = service.stream_generate_content("gemini-pro", request, "test_api_key")
        chunks = [chunk async for chunk in stream]
        mock_stream_generate_content.assert_called_once()
        assert len(chunks) > 0


@pytest.mark.asyncio
async def test_vertex_express_chat_service_generate_content_failure(mock_key_manager):
    """Test the GeminiChatService (from vertex file) generate_content method when the API call fails."""
    with patch("app.service.chat.vertex_express_chat_service.GeminiApiClient.generate_content", new_callable=AsyncMock) as mock_generate_content:
        mock_generate_content.side_effect = Exception(500, "Internal Server Error")
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
        with pytest.raises(Exception):
            await service.generate_content("gemini-pro", request, "test_api_key")
        mock_generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_vertex_express_chat_service_stream_content_failure(mock_key_manager):
    """Test the GeminiChatService (from vertex file) stream_generate_content method when the stream fails."""
    async def mock_stream():
        raise Exception(500, "Internal Server Error")
        yield # this will never be reached

    with patch("app.service.chat.vertex_express_chat_service.GeminiApiClient.stream_generate_content") as mock_stream_generate_content, \
         patch("app.config.config.settings.MAX_RETRIES", 1):
        mock_stream_generate_content.return_value = mock_stream()
        mock_key_manager.handle_api_failure.return_value = None
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
        stream = service.stream_generate_content("gemini-pro", request, "test_api_key")
        with pytest.raises(Exception):
            _ = [chunk async for chunk in stream]
        mock_stream_generate_content.assert_called()
        mock_key_manager.handle_api_failure.assert_called()

def test_has_image_parts():
    assert _has_image_parts([{"parts": [{"image_url": "..."}]}]) is True
    assert _has_image_parts([{"parts": [{"inline_data": "..."}]}]) is True
    assert _has_image_parts([{"parts": [{"text": "..."}]}]) is False
    assert _has_image_parts([]) is False


def test_clean_json_schema_properties():
    schema = {
        "type": "object",
        "properties": {
            "foo": {"type": "string", "description": "bar"}
        },
        "exclusiveMaximum": 10,
        "$schema": "http://json-schema.org/draft-07/schema#",
    }
    cleaned_schema = _clean_json_schema_properties(schema)
    assert "exclusiveMaximum" not in cleaned_schema
    assert "$schema" not in cleaned_schema
    assert "properties" in cleaned_schema

@pytest.mark.parametrize(
    "model, expected_real_model",
    [
        ("gemini-pro-search", "gemini-pro"),
        ("gemini-pro-image", "gemini-pro"),
        ("gemini-pro-non-thinking", "gemini-pro"),
        ("gemini-pro-search-non-thinking", "gemini-pro"),
        ("gemini-pro", "gemini-pro"),
    ]
)
def test_get_real_model(model, expected_real_model):
    assert _get_real_model(model) == expected_real_model
    # assert _get_real_model("gemini-pro-image") == "gemini-pro"
    # assert _get_real_model("gemini-pro-non-thinking") == "gemini-pro"
    # assert _get_real_model("gemini-pro-search-non-thinking") == "gemini-pro"
    # assert _get_real_model("gemini-pro") == "gemini-pro"


def test_get_safety_settings():
    assert _get_safety_settings("gemini-2.0-flash-exp") == GEMINI_2_FLASH_EXP_SAFETY_SETTINGS
    assert _get_safety_settings("gemini-pro") == settings.SAFETY_SETTINGS

def test_build_tools(monkeypatch):
    monkeypatch.setattr("app.service.chat.vertex_express_chat_service.settings.TOOLS_CODE_EXECUTION_ENABLED", True)
    monkeypatch.setattr("app.service.chat.vertex_express_chat_service.settings.URL_CONTEXT_ENABLED", True)
    monkeypatch.setattr("app.service.chat.vertex_express_chat_service.settings.URL_CONTEXT_MODELS", ["gemini-pro"])

    # Test with code execution
    payload = {"contents": []}
    tools = _build_tools("gemini-pro", payload)
    assert "codeExecution" in tools[0]

    # Test with google search
    tools = _build_tools("gemini-pro-search", payload)
    assert "googleSearch" in tools[0]

    # Test with url context
    tools = _build_tools("gemini-pro", payload)
    assert "urlContext" in tools[0]

    # Test with function call
    payload_with_func = {"contents": [{"parts": [{"functionCall": {}}]}]}
    tools = _build_tools("gemini-pro", payload_with_func)
    assert not tools or "googleSearch" not in tools[0]

    # Test with structured output
    payload_structured = {"generationConfig": {"responseMimeType": "application/json"}}
    tools = _build_tools("gemini-pro", payload_structured)
    assert not tools

def test_build_payload():
    request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
    payload = _build_payload("gemini-pro", request)
    assert "contents" in payload
    assert "tools" in payload
    assert "safetySettings" in payload
    assert "generationConfig" in payload

def test_extract_text_from_response(mock_key_manager):
    response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    service = GeminiChatService("http://base.url", mock_key_manager)
    assert service._extract_text_from_response(response) == "Hello"
    assert service._extract_text_from_response({}) == ""
    assert service._extract_text_from_response({"candidates": []}) == ""

def test_create_char_response(mock_key_manager):
    original_response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    service = GeminiChatService("http://base.url", mock_key_manager)
    new_response = service._create_char_response(original_response, "Hi")
    assert new_response["candidates"][0]["content"]["parts"][0]["text"] == "Hi"