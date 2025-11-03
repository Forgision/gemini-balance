import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock
from app.service.chat.gemini_chat_service import (
    GeminiChatService,
    _has_image_parts,
    _extract_file_references,
    _clean_json_schema_properties,
    _build_tools,
    _get_real_model,
    _get_safety_settings,
    _filter_empty_parts,
    _build_payload,
)
from app.domain.gemini_models import GeminiRequest, GeminiContent
from app.config.config import settings
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("API_KEYS")

# Tests for GeminiChatService class

@pytest.mark.asyncio
async def test_gemini_chat_service_generate_content(mock_key_manager):
    """Test the GeminiChatService.generate_content method."""
    with patch("app.service.chat.gemini_chat_service.GeminiApiClient.generate_content", new_callable=AsyncMock) as mock_generate_content:
        mock_generate_content.return_value = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
        response = await service.generate_content("gemini-pro", request, "test_api_key")
        mock_generate_content.assert_called_once()
        assert response["candidates"][0]["content"]["parts"][0]["text"] == "Hello"

@pytest.mark.asyncio
async def test_gemini_chat_service_stream_generate_content(mock_key_manager):
    """Test the GeminiChatService.stream_generate_content method."""
    async def mock_stream():
        yield 'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}'

    with patch("app.service.chat.gemini_chat_service.GeminiApiClient.stream_generate_content") as mock_stream_generate_content:
        mock_stream_generate_content.return_value = mock_stream()
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
        stream = service.stream_generate_content("gemini-pro", request, "test_api_key")
        chunks = [chunk async for chunk in stream]
        mock_stream_generate_content.assert_called_once()
        assert len(chunks) > 0

@pytest.mark.asyncio
async def test_gemini_chat_service_generate_content_no_response(mock_key_manager):
    """Test the GeminiChatService.generate_content method when there is no response."""
    with patch("app.service.chat.gemini_chat_service.GeminiApiClient.generate_content", new_callable=AsyncMock) as mock_generate_content:
        mock_generate_content.side_effect = Exception("Internal Server Error")
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
        with pytest.raises(Exception):
            await service.generate_content("gemini-pro", request, "test_api_key")
        mock_generate_content.assert_called_once()

@pytest.mark.asyncio
async def test_gemini_chat_service_stream_generate_content_failure(mock_key_manager):
    """Test the GeminiChatService.stream_generate_content method when the stream fails."""
    async def mock_stream():
        raise Exception("Internal Server Error")
        yield # this will never be reached

    with patch("app.service.chat.gemini_chat_service.GeminiApiClient.stream_generate_content") as mock_stream_generate_content:
        mock_stream_generate_content.side_effect = Exception("Internal Server Error")
        # mock_stream_generate_content.return_value = mock_stream()
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
        stream = service.stream_generate_content("gemini-pro", request, "test_api_key")
        with pytest.raises(Exception):
            _ = [chunk async for chunk in stream]
        mock_stream_generate_content.assert_called()

@pytest.mark.asyncio
async def test_gemini_chat_service_count_tokens(mock_key_manager):
    """Test the GeminiChatService.count_tokens method."""
    with patch("app.service.chat.gemini_chat_service.GeminiApiClient.count_tokens", new_callable=AsyncMock) as mock_count_tokens:
        mock_count_tokens.return_value = {"totalTokens": 1}
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
        response = await service.count_tokens("gemini-pro", request, "test_api_key")
        mock_count_tokens.assert_called()
        assert response["totalTokens"] == 1

# Tests for helper functions

def test_has_image_parts():
    """Test the _has_image_parts helper function."""
    assert _has_image_parts([{"parts": [{"image_url": "..."}]}]) is True
    assert _has_image_parts([{"parts": [{"inline_data": "..."}]}]) is True
    assert _has_image_parts([{"parts": [{"text": "Hello"}]}]) is False
    assert _has_image_parts([]) is False
    assert _has_image_parts([{"parts": []}]) is False
    assert _has_image_parts([{}]) is False

def test_extract_file_references(monkeypatch):
    """Test the _extract_file_references helper function."""
    monkeypatch.setattr(settings, "BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    contents = [
        {"parts": [{"fileData": {"fileUri": "https://generativelanguage.googleapis.com/v1beta/files/file1"}}]},
        {"parts": [{"fileData": {"fileUri": "https://generativelanguage.googleapis.com/v1beta/files/file2"}}]},
        {"parts": [{"fileData": {"fileUri": "invalid/uri"}}]},
        {"parts": [{"text": "some text"}]},
    ]
    assert _extract_file_references(contents) == ["files/file1", "files/file2"]
    assert _extract_file_references([]) == []
    assert _extract_file_references([{"parts": [{"text": "no file"}]}]) == []

def test_clean_json_schema_properties():
    """Test the _clean_json_schema_properties helper function."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "exclusiveMaximum": 100,
        "$schema": "http://json-schema.org/draft-07/schema#",
        "nested": {
            "const": "value",
            "valid": "field"
        }
    }
    cleaned_schema = _clean_json_schema_properties(schema)
    assert "exclusiveMaximum" not in cleaned_schema
    assert "$schema" not in cleaned_schema
    assert "const" not in cleaned_schema["nested"]
    assert "valid" in cleaned_schema["nested"]
    assert cleaned_schema["type"] == "object"

@pytest.mark.parametrize("model, expected_real_model", [
    ("gemini-pro-search", "gemini-pro"),
    ("gemini-pro-image", "gemini-pro"),
    ("gemini-pro-non-thinking", "gemini-pro"),
    ("gemini-pro-search-non-thinking", "gemini-pro"),
    ("gemini-1.5-pro", "gemini-1.5-pro"),
])
def test_get_real_model(model, expected_real_model):
    """Test the _get_real_model helper function."""
    assert _get_real_model(model) == expected_real_model

def test_get_safety_settings(monkeypatch):
    """Test the _get_safety_settings helper function."""
    custom_settings = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}]
    monkeypatch.setattr(settings, "SAFETY_SETTINGS", custom_settings)
    assert _get_safety_settings("gemini-pro") == custom_settings
    # Assuming GEMINI_2_FLASH_EXP_SAFETY_SETTINGS is different
    assert _get_safety_settings("gemini-2.0-flash-exp") != custom_settings

def test_filter_empty_parts():
    """Test the _filter_empty_parts helper function."""
    contents = [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": []},
        {"role": "user", "parts": [{}]},
        {"role": "user", "parts": ["not a dict"]},
        {"role": "model"},
        None
    ]
    filtered = _filter_empty_parts(contents)
    assert len(filtered) == 1
    assert filtered[0]["parts"][0]["text"] == "Hello"
    assert _filter_empty_parts([]) == []

def test_build_tools(monkeypatch):
    """Test the _build_tools helper function."""
    # Test code execution enabled
    monkeypatch.setattr(settings, "TOOLS_CODE_EXECUTION_ENABLED", True)
    payload = {"contents": [{"parts": [{"text": "code"}]}]}
    tools = _build_tools("gemini-pro", payload)
    assert "codeExecution" in tools[0]

    # Test search model
    tools = _build_tools("gemini-pro-search", payload)
    assert "googleSearch" in tools[0]

    # Test function call disables other tools
    payload_fc = {"contents": [{"parts": [{"functionCall": {}}]}]}
    tools = _build_tools("gemini-pro-search", payload_fc)
    assert not tools or "googleSearch" not in tools[0]

    # Test structured output disables tools
    payload_json = {"generationConfig": {"responseMimeType": "application/json"}}
    tools = _build_tools("gemini-pro", payload_json)
    assert not tools

    # Test merging tools
    payload_merge = {"tools": [{"functionDeclarations": [{"name": "func1"}]}, {"functionDeclarations": [{"name": "func2"}]}]}
    tools = _build_tools("gemini-pro", payload_merge)
    assert len(tools[0]["functionDeclarations"]) == 2

def test_build_payload_non_tts():
    """Test _build_payload for a non-TTS model."""
    request = GeminiRequest(
        contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])],
        generationConfig={"temperature": 0.9}
    )
    payload = _build_payload("gemini-pro", request)
    assert "contents" in payload
    assert "tools" in payload
    assert "safetySettings" in payload
    assert payload["generationConfig"]["temperature"] == 0.9

def test_build_payload_tts():
    """Test _build_payload for a TTS model."""
    request = GeminiRequest(
        contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])],
        systemInstruction={"parts": [{"text": "You are a helpful assistant."}]}
    )
    payload = _build_payload("gemini-tts", request)
    assert "contents" in payload
    assert "systemInstruction" in payload
    assert "tools" not in payload
    assert "safetySettings" not in payload

def test_build_payload_image_model():
    """Test _build_payload for an image generation model."""
    request = GeminiRequest(
        contents=[GeminiContent(role="user", parts=[{"text": "A cat"}])],
        systemInstruction={"parts": [{"text": "This should be removed."}]}
    )
    payload = _build_payload("gemini-pro-image", request)
    assert "systemInstruction" not in payload
    assert payload["generationConfig"]["responseModalities"] == ["Text", "Image"]

def test_build_payload_thinking_config(monkeypatch):
    """Test _build_payload for thinking configuration."""
    monkeypatch.setattr(settings, "SHOW_THINKING_PROCESS", True)
    monkeypatch.setattr(settings, "THINKING_BUDGET_MAP", {"gemini-pro": 1234})
    request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Think about it"}])])

    # Test default thinking config
    payload = _build_payload("gemini-pro", request)
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1234, "includeThoughts": True}

    # Test non-thinking model
    payload_non_thinking = _build_payload("gemini-pro-non-thinking", request)
    assert payload_non_thinking["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}

    # Test client-provided thinking config
    request_with_config = GeminiRequest(
        contents=[GeminiContent(role="user", parts=[{"text": "Think about it"}])],
        generationConfig={"thinkingConfig": {"thinkingBudget": 500}}
    )
    payload_client = _build_payload("gemini-pro", request_with_config)
    assert payload_client["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 500}
