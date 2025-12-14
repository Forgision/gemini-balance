from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# Helper for async generator mock
async def async_gen(items):
    for item in items:
        yield item


@pytest.fixture(autouse=True, scope="session")
def mock_external_apis():
    """
    Mock all external API calls.
    """
    with (
        patch("app.service.client.api_client.GeminiApiClient") as MockGemini,
        patch("app.service.client.api_client.OpenaiApiClient") as MockOpenAI,
        patch("app.service.chat.gemini_chat_service.GeminiApiClient") as MockGeminiChat,
        patch("app.service.model.model_service.GeminiApiClient") as MockGeminiModel,
        patch(
            "app.service.embedding.gemini_embedding_service.GeminiApiClient"
        ) as MockGeminiEmbedding,
        patch("app.service.files.files_service.GeminiApiClient") as MockGeminiFiles,
    ):
        # Configure the main mock (MockGemini) because others are just references to it or new mocks?
        # If we patch the CLASS, we want all patches to return the SAME mock instance or behave similarly.
        # Actually, patch() replaces the class with a MagicMock.
        # We need to configure each mock return value, OR make them all use the same mock.

        # Strategy: Configure MockGemini.return_value.
        # But patching creates DIFFERENT mock objects for each patch call unless we redirect them.
        # However, pytest-mock might allow side_effect or new argument.

        # Simpler: Configure ALL of them.
        mocks = [
            MockGemini,
            MockGeminiChat,
            MockGeminiModel,
            MockGeminiEmbedding,
            MockGeminiFiles,
        ]

        # Setup specific methods to be AsyncMock
        gemini_instance = MockGemini.return_value

        # Make all patches return the same instance
        for m in mocks[1:]:
            m.return_value = gemini_instance

        gemini_instance.generate_content = AsyncMock(
            return_value={
                "candidates": [
                    {"content": {"parts": [{"text": "Mocked Gemini Response"}]}}
                ]
            }
        )
        gemini_instance.stream_generate_content = MagicMock(
            return_value=async_gen(
                [
                    'data: {"candidates": [{"content": {"parts": [{"text": "Mocked "}]}}]}',
                    'data: {"candidates": [{"content": {"parts": [{"text": "Stream"}]}}]}',
                ]
            )
        )
        gemini_instance.count_tokens = AsyncMock(return_value={"totalTokens": 10})
        gemini_instance.embed_content = AsyncMock(
            return_value={"embedding": {"values": [0.1, 0.2, 0.3]}}
        )
        gemini_instance.batch_embed_contents = AsyncMock(
            return_value={
                "embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]
            }
        )
        gemini_instance.get_models = AsyncMock(
            return_value={
                "models": [
                    {"name": "models/gemini-pro", "displayName": "Gemini Pro"},
                    {
                        "name": "models/gemini-pro-vision",
                        "displayName": "Gemini Pro Vision",
                    },
                ]
            }
        )

        openai_instance = MockOpenAI.return_value
        openai_instance.generate_content = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "Mocked OpenAI Response"}}]
            }
        )
        openai_instance.stream_generate_content = MagicMock(
            return_value=async_gen(["Mocked", " OpenAI Stream"])
        )
        openai_instance.create_embeddings = AsyncMock(
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        )
        openai_instance.generate_images = AsyncMock(
            return_value={"data": [{"url": "http://mock-image.url"}]}
        )

        yield {"gemini": gemini_instance, "openai": openai_instance}
