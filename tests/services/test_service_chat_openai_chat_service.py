import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.service.chat.openai_chat_service import OpenAIChatService
from app.domain.openai_models import ChatRequest, ImageGenerationRequest


@pytest.mark.asyncio
async def test_openai_chat_service_generate_content(mock_key_manager):
    """Test the OpenAIChatService.generate_content method."""
    with patch("app.service.chat.openai_chat_service.GeminiApiClient.generate_content", new_callable=AsyncMock) as mock_generate_content:
        gemini_response = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 2,
                "totalTokenCount": 3
            }
        }
        mock_generate_content.return_value = gemini_response
        service = OpenAIChatService("http://base.url", mock_key_manager)
        request = ChatRequest(messages=[{"role": "user", "content": "Hello"}], stream=False)
        response = await service.create_chat_completion(request, "test_api_key")
        mock_generate_content.assert_called_once()
        assert response is not None
        assert isinstance(response, dict)
        assert response["choices"][0]["message"]["content"] == "Hello"


@pytest.mark.asyncio
async def test_openai_chat_service_stream_content(mock_key_manager):
    """Test the OpenAIChatService.stream_generate_content method."""
    async def mock_stream():
        yield 'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}'

    with patch("app.service.chat.openai_chat_service.GeminiApiClient.stream_generate_content") as mock_stream_generate_content:
        mock_stream_generate_content.return_value = mock_stream()
        service = OpenAIChatService("http://base.url", mock_key_manager)
        request = ChatRequest(messages=[{"role": "user", "content": "Hello"}], stream=True)
        stream = service.create_chat_completion(request, "test_api_key")
        assert stream is not None
        assert isinstance(stream, AsyncMock)
        chunks = [chunk async for chunk in stream]
        mock_stream_generate_content.assert_called_once()
        assert len(chunks) > 0


@pytest.mark.asyncio
async def test_openai_chat_service_generate_content_failure(mock_key_manager):
    """Test the OpenAIChatService.generate_content method when the API call fails."""
    with patch("app.service.chat.openai_chat_service.GeminiApiClient.generate_content", new_callable=AsyncMock) as mock_generate_content:
        mock_generate_content.side_effect = Exception(500, "Internal Server Error")
        service = OpenAIChatService("http://base.url", mock_key_manager)
        request = ChatRequest(messages=[{"role": "user", "content": "Hello"}])
        with pytest.raises(Exception):
            await service.create_chat_completion(request, "test_api_key")
        mock_generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_openai_chat_service_stream_content_failure(mock_key_manager):
    """Test the OpenAIChatService.stream_generate_content method when the stream fails."""
    async def mock_stream():
        raise Exception(500, "Internal Server Error")
        yield # this will never be reached

    with patch("app.service.chat.openai_chat_service.GeminiApiClient.stream_generate_content") as mock_stream_generate_content:
        mock_stream_generate_content.return_value = mock_stream()
        service = OpenAIChatService("http://base.url", mock_key_manager)
        request = ChatRequest(messages=[{"role": "user", "content": "Hello"}], stream=True)
        stream = await service.create_chat_completion(request, "test_api_key")
        with pytest.raises(Exception):
            assert stream is not None
            assert isinstance(stream, AsyncMock)
            _ = [chunk async for chunk in stream]
        mock_stream_generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_openai_chat_service_image_generation(mock_key_manager):
    """Test the OpenAIChatService.create_image_chat_completion method."""
    with patch("app.service.image.image_create_service.ImageCreateService.generate_images_chat") as mock_image_generation:
        mock_image_generation.return_value = "http://image.url"
        service = OpenAIChatService("http://base.url", mock_key_manager)
        request = ChatRequest(messages=[{"role": "user", "content": "A cat"}], model="imagen-3.0-generate-002", stream=False)
        response = await service.create_image_chat_completion(request, "test_api_key")
        mock_image_generation.assert_called_once()
        assert response is not None
        assert isinstance(response, dict)
        assert "http://image.url" in response["choices"][0]["message"]["content"]
