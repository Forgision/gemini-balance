import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.service.chat.gemini_chat_service import GeminiChatService
from app.domain.gemini_models import GeminiRequest, GeminiContent

@pytest.fixture
def mock_key_manager():
    """Fixture for KeyManager."""
    return MagicMock()

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
        mock_stream_generate_content.return_value = mock_stream()
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
        stream = service.stream_generate_content("gemini-pro", request, "test_api_key")
        with pytest.raises(Exception):
            _ = [chunk async for chunk in stream]
        mock_stream_generate_content.assert_called_once()

@pytest.mark.asyncio
async def test_gemini_chat_service_count_tokens(mock_key_manager):
    """Test the GeminiChatService.count_tokens method."""
    with patch("app.service.chat.gemini_chat_service.GeminiApiClient.count_tokens", new_callable=AsyncMock) as mock_count_tokens:
        mock_count_tokens.return_value = {"totalTokens": 1}
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "Hello"}])])
        response = await service.count_tokens("gemini-pro", request, "test_api_key")
        mock_count_tokens.assert_called_once()
        assert response["totalTokens"] == 1
