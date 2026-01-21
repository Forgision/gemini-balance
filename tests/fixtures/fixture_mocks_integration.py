import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from tests.fixtures.fixture_consts import TEST_AUTH_TOKEN

# Helper for async generator mock
async def async_gen(items):
    for item in items:
        yield item

@pytest.fixture(scope="session")
def mock_gemini_api_client():
    """Session-scoped mock GeminiApiClient with realistic responses."""

    async def generate_content_side_effect(payload, model, api_key):
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "This is a mock response from Gemini API."}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ]
        }

    async def stream_generate_content_side_effect(payload, model, api_key):
        chunks = [
            'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}\n\n',
            'data: {"candidates":[{"content":{"parts":[{"text":" "}],"role":"model"}}]}\n\n',
            'data: {"candidates":[{"content":{"parts":[{"text":"world"}],"role":"model"}}]}\n\n',
            'data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP"}]}\n\n',
        ]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.001)

    async def count_tokens_side_effect(payload, model, api_key):
        return {"totalTokens": 42}

    async def embed_content_side_effect(payload, model, api_key):
        return {
            "embedding": {
                "values": [0.1] * 768
            }
        }

    async def batch_embed_contents_side_effect(payload, model, api_key):
        return {"embeddings": [{"values": [0.1] * 768}, {"values": [0.2] * 768}]}

    async def get_models_side_effect(api_key):
        return {
            "models": [
                {
                    "name": "models/gemini-pro",
                    "displayName": "Gemini Pro",
                    "description": "Best model for general tasks",
                    "supportedGenerationMethods": [
                        "generateContent",
                        "streamGenerateContent",
                    ],
                },
                {
                    "name": "models/gemini-2.0-flash-exp",
                    "displayName": "Gemini 2.0 Flash Experimental",
                    "description": "Fast experimental model",
                    "supportedGenerationMethods": [
                        "generateContent",
                        "streamGenerateContent",
                    ],
                },
            ]
        }

    mock = MagicMock()
    mock.generate_content = AsyncMock(side_effect=generate_content_side_effect)
    mock.count_tokens = AsyncMock(side_effect=count_tokens_side_effect)
    mock.embed_content = AsyncMock(side_effect=embed_content_side_effect)
    mock.batch_embed_contents = AsyncMock(side_effect=batch_embed_contents_side_effect)
    mock.get_models = AsyncMock(side_effect=get_models_side_effect)

    async def stream_generator(payload, model, api_key):
        async for chunk in stream_generate_content_side_effect(payload, model, api_key):
            yield chunk

    mock.stream_generate_content = stream_generator

    return mock


@pytest.fixture(scope="session")
def mock_openai_api_client():
    """Session-scoped mock OpenaiApiClient with realistic responses."""

    async def generate_content_side_effect(payload, model, api_key):
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a mock response from OpenAI API.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    async def stream_generate_content_side_effect(payload, model, api_key):
        chunks = [
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"'
            + model
            + '","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"'
            + model
            + '","choices":[{"index":0,"delta":{"content":" "},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"'
            + model
            + '","choices":[{"index":0,"delta":{"content":"world"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"'
            + model
            + '","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.001)

    async def get_models_side_effect(api_key):
        return {
            "data": [
                {
                    "id": "gemini-pro",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "google",
                },
                {
                    "id": "gemini-2.0-flash-exp",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "google",
                },
            ]
        }

    async def create_embeddings_side_effect(input, model, api_key):
        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1] * 768, "index": 0}],
            "model": model,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }

    async def generate_images_side_effect(payload, api_key):
        return {
            "created": 1234567890,
            "data": [
                {
                    "url": "https://example.com/image.png",
                    "revised_prompt": payload.get("prompt", ""),
                }
            ],
        }

    mock = MagicMock()
    mock.generate_content = AsyncMock(side_effect=generate_content_side_effect)

    async def stream_generator(payload, model, api_key):
        async for chunk in stream_generate_content_side_effect(payload, model, api_key):
            yield chunk

    mock.stream_generate_content = stream_generator
    mock.get_models = AsyncMock(side_effect=get_models_side_effect)
    mock.create_embeddings = AsyncMock(side_effect=create_embeddings_side_effect)
    mock.generate_images = AsyncMock(side_effect=generate_images_side_effect)

    return mock


@pytest.fixture(scope="function")
def patched_service_clients(monkeypatch):
    """Function-scoped fixture to patch OpenAI and Gemini clients used directly by services."""
    import openai
    from google import genai

    mock_openai_client = MagicMock()

    def create_embedding_side_effect(input, model):
        from openai.types import CreateEmbeddingResponse
        from openai.types.create_embedding_response import Usage
        from openai.types import Embedding

        num_items = len(input) if isinstance(input, list) else 1

        embedding_data = [
            Embedding(object="embedding", index=i, embedding=[0.1] * 768)
            for i in range(num_items)
        ]

        usage_obj = Usage(prompt_tokens=10, total_tokens=10)

        return CreateEmbeddingResponse(
            object="list", data=embedding_data, model=model, usage=usage_obj
        )

    mock_openai_client.embeddings.create = MagicMock(
        side_effect=create_embedding_side_effect
    )

    from google.genai import types

    mock_genai_client = MagicMock()

    def generate_images_side_effect(model, prompt, config):
        mock_generated_image = types.GeneratedImage()
        mock_image_obj = MagicMock()
        mock_image_obj.image_bytes = b"fake_image_data"
        mock_generated_image.image = mock_image_obj

        mock_response = MagicMock()
        mock_response.generated_images = [mock_generated_image]
        return mock_response

    mock_genai_client.models.generate_images = MagicMock(
        side_effect=generate_images_side_effect
    )

    monkeypatch.setattr(openai, "OpenAI", lambda *args, **kwargs: mock_openai_client)
    monkeypatch.setattr(genai, "Client", lambda *args, **kwargs: mock_genai_client)

    return {"openai_client": mock_openai_client, "genai_client": mock_genai_client}


@pytest.fixture(scope="function")
def patched_api_clients(mock_gemini_api_client, mock_openai_api_client):
    """Function-scoped fixture to patch API clients in the application."""

    def gemini_factory(*args, **kwargs):
        return mock_gemini_api_client

    def openai_factory(*args, **kwargs):
        return mock_openai_api_client

    with (
        patch("app.service.client.api_client.GeminiApiClient", new=gemini_factory),
        patch("app.service.client.api_client.OpenaiApiClient", new=openai_factory),
        patch(
            "app.service.chat.gemini_chat_service.GeminiApiClient", new=gemini_factory
        ),
        patch(
            "app.service.chat.openai_chat_service.GeminiApiClient", new=gemini_factory
        ),
        patch(
            "app.service.chat.vertex_express_chat_service.GeminiApiClient",
            new=gemini_factory,
        ),
        patch(
            "app.service.openai_compatiable.openai_compatiable_service.OpenaiApiClient",
            new=openai_factory,
        ),
        patch(
            "app.service.embedding.gemini_embedding_service.GeminiApiClient",
            new=gemini_factory,
        ),
        patch("app.service.files.files_service.GeminiApiClient", new=gemini_factory),
        patch("app.service.model.model_service.GeminiApiClient", new=gemini_factory),
        patch("app.service.claude_proxy_service.GeminiApiClient", new=gemini_factory),
    ):
        yield {"gemini": mock_gemini_api_client, "openai": mock_openai_api_client}
