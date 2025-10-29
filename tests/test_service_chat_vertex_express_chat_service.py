import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.service.chat.vertex_express_chat_service import GeminiChatService
from app.domain.gemini_models import GeminiRequest


@pytest.fixture
def mock_key_manager():
    """Fixture for KeyManager."""
    mock = MagicMock()
    mock.get_key = AsyncMock(return_value="test_api_key")
    return mock


@pytest.mark.asyncio
async def test_vertex_express_chat_service_generate_content(mock_key_manager):
    """Test the GeminiChatService (from vertex file) generate_content method."""
    with patch("app.service.chat.vertex_express_chat_service.GeminiApiClient.generate_content", new_callable=AsyncMock) as mock_generate_content:
        mock_generate_content.return_value = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[{"role": "user", "parts": [{"text": "Hello"}]}])
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
        request = GeminiRequest(contents=[{"role": "user", "parts": [{"text": "Hello"}]}])
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
        request = GeminiRequest(contents=[{"role": "user", "parts": [{"text": "Hello"}]}])
        with pytest.raises(Exception):
            await service.generate_content("gemini-pro", request, "test_api_key")
        mock_generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_vertex_express_chat_service_stream_content_failure(mock_key_manager):
    """Test the GeminiChatService (from vertex file) stream_generate_content method when the stream fails."""
    async def mock_stream():
        raise Exception(500, "Internal Server Error")
        yield # this will never be reached

    with patch("app.service.chat.vertex_express_chat_service.GeminiApiClient.stream_generate_content") as mock_stream_generate_content:
        mock_stream_generate_content.return_value = mock_stream()
        service = GeminiChatService("http://base.url", mock_key_manager)
        request = GeminiRequest(contents=[{"role": "user", "parts": [{"text": "Hello"}]}])
        stream = service.stream_generate_content("gemini-pro", request, "test_api_key")
        with pytest.raises(Exception):
            _ = [chunk async for chunk in stream]
        mock_stream_generate_content.assert_called_once()
