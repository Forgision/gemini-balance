"""
Configuration file for tests_no_api integration tests.
Provides fixtures for in-memory databases, mocked API clients, and test application setup.
"""

import pytest
import pytest_asyncio
import asyncio
import json
import datetime
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from pytest import MonkeyPatch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from sqlalchemy import select, insert, update, inspect as sqlalchemy_inspect
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from app.config.config import settings
from app.service.key.key_manager import KeyManager
from app.dependencies import get_key_manager


# Test constants
TEST_API_KEYS = ["test_key_1", "test_key_2", "test_key_3"]
TEST_VERTEX_API_KEYS = ["test_vertex_key_1", "test_vertex_key_2"]
TEST_AUTH_TOKEN = "test_auth_token_12345"
TEST_ALLOWED_TOKENS = ["test_token_1", "test_token_2"]


def run_async_safe(coro):
    """
    Run an async coroutine safely, handling both cases:
    - When there's a running event loop (e.g., in async tests)
    - When there's no running event loop (e.g., in session fixtures)
    """
    try:
        # Try to get the running event loop
        asyncio.get_running_loop()
        # If we're here, there's a running loop
        # We need to run this in a new thread with a new event loop
        import concurrent.futures

        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        # No running event loop, safe to use asyncio.run()
        return asyncio.run(coro)


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch fixture for database settings."""
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def patch_database_settings(monkeypatch_session):
    """Patch database settings to use in-memory SQLite for both main DB and KeyManager DB."""
    # Patch main database
    monkeypatch_session.setattr(settings, "DATABASE_TYPE", "sqlite")
    monkeypatch_session.setattr(settings, "SQLITE_DATABASE", "integration_test.sqlite")

    # Patch test configuration
    monkeypatch_session.setattr(settings, "API_KEYS", TEST_API_KEYS)
    monkeypatch_session.setattr(settings, "VERTEX_API_KEYS", TEST_VERTEX_API_KEYS)
    monkeypatch_session.setattr(settings, "AUTH_TOKEN", TEST_AUTH_TOKEN)
    monkeypatch_session.setattr(settings, "ALLOWED_TOKENS", TEST_ALLOWED_TOKENS)

    # Do not reload db_conn as it breaks Base class identity
    # importlib.reload(db_conn)

    # Reload KeyManager module to apply new DB URL

    # Do not reload key_manager either
    # importlib.reload(km)


@pytest.fixture(scope="session")
def in_memory_db_engine(patch_database_settings, monkeypatch_session):
    """Session-scoped fixture that patches the app to use a shared async in-memory SQLite engine."""
    from app.database.connection import Base
    import app.database.connection as db_conn

    db_dir = Path("data")
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "integration_test.sqlite"
    if db_path.exists():
        db_path.unlink()

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    run_async_safe(create_tables())

    session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    # Patch the application-level engine/session so all code paths share the same DB
    monkeypatch_session.setattr(db_conn, "engine", engine, raising=False)
    monkeypatch_session.setattr(
        db_conn, "AsyncSessionLocal", session_maker, raising=False
    )

    # Dynamic patching: Iterate over all loaded modules and patch AsyncSessionLocal and engine
    # This ensures that any module that imported them at top level gets the patched version
    import sys

    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("app."):
            if hasattr(module, "AsyncSessionLocal"):
                monkeypatch_session.setattr(
                    module, "AsyncSessionLocal", session_maker, raising=False
                )
            if hasattr(module, "engine"):
                monkeypatch_session.setattr(module, "engine", engine, raising=False)

    yield engine

    async def dispose_engine():
        await engine.dispose()

    run_async_safe(dispose_engine())
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(scope="session")
def key_manager_async_engine(patch_database_settings):
    """Session-scoped fixture to create in-memory async SQLite engine for KeyManager."""
    from app.service.key.key_manager import Base

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    # Create tables
    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    run_async_safe(create_tables())

    yield engine

    # Cleanup
    async def dispose_engine():
        await engine.dispose(close=True)

    run_async_safe(dispose_engine())


@pytest.fixture(scope="session")
def mock_gemini_api_client():
    """Session-scoped mock GeminiApiClient with realistic responses."""

    async def generate_content_side_effect(payload, model, api_key):
        """Mock generate_content method."""
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
        """Mock stream_generate_content method - returns SSE-formatted chunks."""
        chunks = [
            'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}\n\n',
            'data: {"candidates":[{"content":{"parts":[{"text":" "}],"role":"model"}}]}\n\n',
            'data: {"candidates":[{"content":{"parts":[{"text":"world"}],"role":"model"}}]}\n\n',
            'data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP"}]}\n\n',
        ]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.001)  # Small delay to make it properly async

    async def count_tokens_side_effect(payload, model, api_key):
        """Mock count_tokens method."""
        return {"totalTokens": 42}

    async def embed_content_side_effect(payload, model, api_key):
        """Mock embed_content method."""
        return {
            "embedding": {
                "values": [0.1] * 768  # Typical embedding dimension
            }
        }

    async def batch_embed_contents_side_effect(payload, model, api_key):
        """Mock batch_embed_contents method."""
        return {"embeddings": [{"values": [0.1] * 768}, {"values": [0.2] * 768}]}

    async def get_models_side_effect(api_key):
        """Mock get_models method."""
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

    # Fix stream_generate_content to return async generator
    async def stream_generator(payload, model, api_key):
        async for chunk in stream_generate_content_side_effect(payload, model, api_key):
            yield chunk

    mock.stream_generate_content = stream_generator

    return mock


@pytest.fixture(scope="session")
def mock_openai_api_client():
    """Session-scoped mock OpenaiApiClient with realistic responses."""

    async def generate_content_side_effect(payload, model, api_key):
        """Mock generate_content method."""
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
        """Mock stream_generate_content method - returns OpenAI SSE chunks."""
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
            await asyncio.sleep(0.001)  # Small delay to make it properly async

    async def get_models_side_effect(api_key):
        """Mock get_models method."""
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
        """Mock create_embeddings method."""
        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1] * 768, "index": 0}],
            "model": model,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }

    async def generate_images_side_effect(payload, api_key):
        """Mock generate_images method."""
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

    # Fix stream_generate_content to return async generator
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

    # Mock OpenAI client for EmbeddingService
    # CreateEmbeddingResponse is a Pydantic model, but we'll use a mock with model_dump
    mock_openai_client = MagicMock()

    # Create proper mock response that can be serialized by FastAPI
    def create_embedding_side_effect(input, model):
        # Try to use the real CreateEmbeddingResponse type if available
        try:
            from openai.types import CreateEmbeddingResponse
            from openai.types.embeddings import Embedding  # type: ignore

            try:
                from openai.types.embeddings import Usage  # type: ignore
            except ImportError:
                from openai.types import Usage

            num_items = len(input) if isinstance(input, list) else 1
            embedding_list = [
                Embedding(object="embedding", index=i, embedding=[0.1] * 768)
                for i in range(num_items)
            ]
            usage_obj = Usage(prompt_tokens=10, total_tokens=10)
            return CreateEmbeddingResponse(
                object="list", data=embedding_list, model=model, usage=usage_obj
            )
        except (ImportError, AttributeError, TypeError):
            # Fallback: create a Pydantic-like class that FastAPI can serialize
            from pydantic import BaseModel

            class EmbeddingData(BaseModel):
                object: str = "embedding"
                index: int
                embedding: list

            class UsageData(BaseModel):
                prompt_tokens: int = 10
                total_tokens: int = 10

            class CreateEmbeddingResponseModel(BaseModel):
                object: str = "list"
                data: list
                model: str
                usage: UsageData

            num_items = len(input) if isinstance(input, list) else 1
            embedding_list = [
                EmbeddingData(index=i, embedding=[0.1] * 768) for i in range(num_items)
            ]
            usage_obj = UsageData()
            return CreateEmbeddingResponseModel(
                object="list",
                data=[e.model_dump() for e in embedding_list],
                model=model,
                usage=usage_obj,
            )

    mock_openai_client.embeddings.create = MagicMock(
        side_effect=create_embedding_side_effect
    )

    # Mock genai.Client for ImageCreateService
    from google.genai import types

    mock_genai_client = MagicMock()

    def generate_images_side_effect(model, prompt, config):
        # Create proper mock image response
        mock_generated_image = types.GeneratedImage()
        # Set attributes that the service accesses
        mock_image_obj = MagicMock()
        mock_image_obj.image_bytes = b"fake_image_data"
        mock_generated_image.image = mock_image_obj

        mock_response = MagicMock()
        mock_response.generated_images = [mock_generated_image]
        return mock_response

    mock_genai_client.models.generate_images = MagicMock(
        side_effect=generate_images_side_effect
    )

    # Patch OpenAI and Gemini clients
    monkeypatch.setattr(openai, "OpenAI", lambda *args, **kwargs: mock_openai_client)
    monkeypatch.setattr(genai, "Client", lambda *args, **kwargs: mock_genai_client)

    return {"openai_client": mock_openai_client, "genai_client": mock_genai_client}


@pytest.fixture(scope="function")
def patched_api_clients(mock_gemini_api_client, mock_openai_api_client):
    """Function-scoped fixture to patch API clients in the application."""

    # Create factory functions that return our mocks when called
    def gemini_factory(*args, **kwargs):
        return mock_gemini_api_client

    def openai_factory(*args, **kwargs):
        return mock_openai_api_client

    # Patch the classes so that when services instantiate them, they get our mocks
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


@pytest_asyncio.fixture(scope="function")
async def test_key_manager(key_manager_async_engine):
    """Function-scoped fixture to create and initialize a real KeyManager instance."""
    # Create new async session maker for this test
    async_session_maker = async_sessionmaker(
        key_manager_async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    # Create tables for KeyManager database before initializing
    # The key_manager module doesn't have an init_database function,
    # so we create the tables directly using the test engine
    from app.database.connection import Base

    async def create_key_manager_tables():
        """Create KeyManager tables using the test engine."""
        async with key_manager_async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Create tables before KeyManager initialization
    await create_key_manager_tables()

    # Patch asyncio.create_task to ensure tasks are created in the current event loop
    # This prevents "Task got Future attached to a different loop" errors
    original_create_task = asyncio.create_task
    current_loop = asyncio.get_event_loop()

    def patched_create_task(coro, name=None):
        """Create task in the current event loop."""
        try:
            # Ensure we're using the current loop
            loop = asyncio.get_running_loop()
            return loop.create_task(coro, name=name)
        except RuntimeError:
            # If no running loop, use the current loop
            return current_loop.create_task(coro, name=name)

    # Temporarily patch asyncio.create_task
    import asyncio as asyncio_module

    asyncio_module.create_task = patched_create_task

    try:
        # Create KeyManager instance
        key_manager = KeyManager(
            api_keys=TEST_API_KEYS,
            vertex_api_keys=TEST_VERTEX_API_KEYS,
            async_session_maker=async_session_maker,
        )

        # Initialize KeyManager
        # Provide default rate limits to avoid web scraping during tests
        default_rate_limits = {
            "gemini-pro": {"RPM": 60, "TPM": 1000000, "RPD": 1500},
            "gemini-2.0-flash-exp": {"RPM": 15, "TPM": 1000000, "RPD": 1500},
            "gemini-2.5-pro": {"RPM": 60, "TPM": 1000000, "RPD": 1500},
            "gemini-2.5-flash": {"RPM": 60, "TPM": 1000000, "RPD": 1500},
        }

        try:
            await key_manager.init(rate_limit_data=default_rate_limits)
        except Exception:
            # If init fails, try with default
            await key_manager.init()

        # Wait a small amount to ensure initialization is complete
        # This ensures the background worker has started and any initial commits are done
        await asyncio.sleep(0.1)

        yield key_manager

        # Cleanup - ensure background task is properly stopped
        try:
            if key_manager._background_task and not key_manager._background_task.done():
                key_manager._stop_event.set()
                key_manager._background_task.cancel()
                try:
                    await asyncio.wait_for(key_manager._background_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            await key_manager.shutdown()
        except Exception:
            pass
    finally:
        # Restore original functions
        asyncio_module.create_task = original_create_task


@pytest_asyncio.fixture(scope="function")
async def test_app(
    in_memory_db_engine, test_key_manager, patched_api_clients, patched_service_clients
):
    """Function-scoped fixture to create a test FastAPI app using the real create_app() function."""
    # Ensure KeyManager is initialized (it should be from test_key_manager fixture, but verify)
    if not test_key_manager.is_ready:
        await test_key_manager.init()

    # FULLY initialize database with required config BEFORE create_app() runs
    from app.database.connection import AsyncSessionLocal, connect_to_db
    from app.database.initialization import initialize_database
    from app.config.config import sync_initial_settings
    from app.database.models import Settings as SettingsModel

    # Delete old database file if it exists (for SQLite file-based databases)
    # if settings.DATABASE_TYPE == "sqlite" and settings.SQLITE_DATABASE != ":memory:":
    #     db_path = Path("data") / settings.SQLITE_DATABASE
    #     if db_path.exists():
    #         db_path.unlink()

    # Clear all tables to ensure clean state for each test
    # This is necessary because we are reusing the session-scoped engine/database
    from app.database.connection import engine
    from sqlalchemy import text

    async with engine.begin() as conn:
        # Disable foreign key checks to allow truncation
        await conn.execute(text("PRAGMA foreign_keys = OFF"))

        # Get all table names
        tables = await conn.run_sync(
            lambda sync_conn: sqlalchemy_inspect(sync_conn).get_table_names()
        )

        for table in tables:
            # Use DELETE FROM instead of TRUNCATE for SQLite
            await conn.execute(text(f"DELETE FROM {table}"))

        # Re-enable foreign keys
        await conn.execute(text("PRAGMA foreign_keys = ON"))

    # Ensure tables exist (create them if they don't)
    # This handles the case where tables might not exist yet (first run)
    # or if they were somehow dropped
    from app.database.connection import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Step 1: Initialize database (create tables and import settings)
    await initialize_database()

    # Step 2: Connect to database (verify connectivity)
    await connect_to_db()

    # Step 3: Sync initial settings first (this creates t_settings table and syncs config)
    await sync_initial_settings()

    # Step 4: Ensure TEST_AUTH_TOKEN and TEST_ALLOWED_TOKENS are in database AFTER sync
    # This ensures our test tokens override any defaults from sync
    try:
        async with AsyncSessionLocal() as session:
            # Check if AUTH_TOKEN exists in database
            query = select(SettingsModel.key, SettingsModel.value).where(
                SettingsModel.key.in_(["AUTH_TOKEN", "ALLOWED_TOKENS"])
            )
            result = await session.execute(query)
            rows = result.fetchall()
            existing = {
                dict(row._mapping)["key"]: dict(row._mapping)["value"] for row in rows
            }

            now = datetime.datetime.now(datetime.timezone.utc)

            # Insert or update AUTH_TOKEN
            if "AUTH_TOKEN" not in existing:
                query_insert = insert(SettingsModel).values(
                    key="AUTH_TOKEN",
                    value=TEST_AUTH_TOKEN,
                    description="AUTH_TOKEN configuration setting",
                    created_at=now,
                    updated_at=now,
                )
                await session.execute(query_insert)
                await session.commit()
            elif existing["AUTH_TOKEN"] != TEST_AUTH_TOKEN:
                query_update = (
                    update(SettingsModel)
                    .where(SettingsModel.key == "AUTH_TOKEN")
                    .values(value=TEST_AUTH_TOKEN, updated_at=now)
                )
                await session.execute(query_update)
                await session.commit()

            # Insert or update ALLOWED_TOKENS
            allowed_tokens_json = json.dumps(TEST_ALLOWED_TOKENS, ensure_ascii=False)
            if "ALLOWED_TOKENS" not in existing:
                query_insert = insert(SettingsModel).values(
                    key="ALLOWED_TOKENS",
                    value=allowed_tokens_json,
                    description="ALLOWED_TOKENS configuration setting",
                    created_at=now,
                    updated_at=now,
                )
                await session.execute(query_insert)
                await session.commit()
            elif existing["ALLOWED_TOKENS"] != allowed_tokens_json:
                query_update = (
                    update(SettingsModel)
                    .where(SettingsModel.key == "ALLOWED_TOKENS")
                    .values(value=allowed_tokens_json, updated_at=now)
                )
                await session.execute(query_update)
                await session.commit()

        # Re-sync to load our test tokens into memory settings
        await sync_initial_settings()

    except Exception:
        # If database setup fails, ensure settings are still patched
        pass

    # Step 5: Ensure settings object ALWAYS has test tokens (final override)
    settings.AUTH_TOKEN = TEST_AUTH_TOKEN
    settings.ALLOWED_TOKENS = TEST_ALLOWED_TOKENS

    # Create app using the real create_app() function
    from app.core.application import create_app

    # Override lifespan to skip database reinitialization (already done) and use test KeyManager
    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        """Test lifespan that skips database init (already done) and uses test KeyManager."""
        from app.database.connection import disconnect_from_db

        # Database is already initialized and connected, settings are already synced
        # Just ensure test KeyManager is in app.state
        app.state.key_manager = test_key_manager

        yield

        # Cleanup (skip KeyManager shutdown as it's handled by test_key_manager fixture)
        # Only disconnect from database, don't drop tables (in-memory will be gone anyway)
        await disconnect_from_db()

    # Temporarily patch create_app to use test lifespan
    from app.core import application as app_module

    original_lifespan = app_module.lifespan
    app_module.lifespan = test_lifespan

    try:
        # Create the app (this sets up everything including routes, middleware, etc.)
        # The database is already initialized, so lifespan won't reinitialize it
        app = create_app()

        # Ensure KeyManager is set in app.state (lifespan will do this, but ensure it's set)
        app.state.key_manager = test_key_manager
    finally:
        # Restore original lifespan
        app_module.lifespan = original_lifespan

    # Override get_key_manager dependency
    # FastAPI will inject Request, but we ignore it and return the test KeyManager
    from fastapi import Request as FastAPIRequest, Request

    async def override_get_key_manager(request: FastAPIRequest = None):
        return test_key_manager

    app.dependency_overrides[get_key_manager] = override_get_key_manager

    # Mock security dependencies to allow all requests using TEST_AUTH_TOKEN and TEST_ALLOWED_TOKENS
    from app.router import (
        gemini_routes,
        openai_routes,
        openai_compatible_routes,
        claude_routes,
    )
    from app.router import scheduler_routes, stats_routes

    # Patch verify_auth_token function to accept TEST_AUTH_TOKEN only
    # This is for AuthMiddleware which protects website/backend routes
    def mock_verify_auth_token_func(token: str) -> bool:
        """Mock verify_auth_token to accept TEST_AUTH_TOKEN only (for website/backend routes)."""
        # verify_auth_token only checks against AUTH_TOKEN, not ALLOWED_TOKENS
        return token == TEST_AUTH_TOKEN

    # Patch the function in the security module
    import app.core.security as security_module

    original_verify_auth_token = security_module.verify_auth_token
    security_module.verify_auth_token = mock_verify_auth_token_func

    async def mock_verify_key_or_goog_api_key(key=None, x_goog_api_key=None):
        """Mock verify_key_or_goog_api_key to accept TEST_AUTH_TOKEN and TEST_ALLOWED_TOKENS."""
        # If key in URL is provided and valid, return it
        if key is not None and (key == TEST_AUTH_TOKEN or key in TEST_ALLOWED_TOKENS):
            return key
        # Otherwise check x-goog-api-key header
        if x_goog_api_key is not None:
            if (
                x_goog_api_key == TEST_AUTH_TOKEN
                or x_goog_api_key in TEST_ALLOWED_TOKENS
            ):
                return x_goog_api_key
        # If neither is valid, return TEST_AUTH_TOKEN as default (for testing)
        return TEST_AUTH_TOKEN

    async def mock_verify_authorization(authorization=None):
        """Mock verify_authorization to accept TEST_AUTH_TOKEN and TEST_ALLOWED_TOKENS."""
        if authorization and authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
            if token == TEST_AUTH_TOKEN or token in TEST_ALLOWED_TOKENS:
                return token
        # Return TEST_AUTH_TOKEN as default for testing
        return TEST_AUTH_TOKEN

    async def mock_verify_auth_token_header(authorization=None):
        """Mock verify_auth_token (header version) to accept TEST_AUTH_TOKEN."""
        if authorization and authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
            if token == TEST_AUTH_TOKEN:
                return token
        # Return TEST_AUTH_TOKEN as default for testing
        return TEST_AUTH_TOKEN

    async def mock_verify_token_for_dependencies(request: Request):
        """Mock verify_token function used in scheduler_routes and stats_routes."""
        # This dependency checks cookies and calls verify_auth_token
        # Since verify_auth_token is already patched to accept TEST_AUTH_TOKEN,
        # we can just let it run. But to be safe, we'll verify the cookie matches TEST_AUTH_TOKEN
        from app.core.security import verify_auth_token

        auth_token = (
            request.cookies.get("auth_token") if hasattr(request, "cookies") else None
        )
        if auth_token and verify_auth_token(auth_token):
            return  # Pass - don't raise exception
        # If no valid token, raise 401 (shouldn't happen in tests with proper cookies)
        from fastapi import HTTPException
        from starlette import status

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Override security dependencies
    app.dependency_overrides[
        gemini_routes.security_service.verify_key_or_goog_api_key
    ] = mock_verify_key_or_goog_api_key

    app.dependency_overrides[openai_routes.security_service.verify_authorization] = (
        mock_verify_authorization
    )

    app.dependency_overrides[
        openai_compatible_routes.security_service.verify_authorization
    ] = mock_verify_authorization

    app.dependency_overrides[claude_routes.security_service.verify_auth_token] = (
        mock_verify_auth_token_header
    )

    # Override verify_token dependencies for scheduler and stats routes
    if hasattr(scheduler_routes, "verify_token"):
        app.dependency_overrides[scheduler_routes.verify_token] = (
            mock_verify_token_for_dependencies
        )

    if hasattr(stats_routes, "verify_token"):
        app.dependency_overrides[stats_routes.verify_token] = (
            mock_verify_token_for_dependencies
        )

    # Ensure KeyManager is stored in app.state (should already be set, but ensure it)
    app.state.key_manager = test_key_manager

    # Database is already initialized and connected, settings are already synced
    # Just ensure app.state is properly set up
    if not hasattr(app.state, "key_manager"):
        app.state.key_manager = test_key_manager

    # Patch scheduler stop function to handle closed event loop gracefully in tests
    from app.scheduler import scheduled_tasks as scheduler_module
    from app.log.logger import Logger

    scheduler_logger = Logger.setup_logger("scheduler")
    original_stop_scheduler = scheduler_module.stop_scheduler

    def patched_stop_scheduler():
        """Patched stop_scheduler that handles closed event loop gracefully."""
        try:
            return original_stop_scheduler()
        except RuntimeError as e:
            # If event loop is closed, scheduler is already stopped or not running
            if "Event loop is closed" in str(e):
                scheduler_logger.info(
                    "Scheduler event loop is closed (likely already stopped)."
                )
                # Reset scheduler_instance to None to allow restart
                scheduler_module.scheduler_instance = None
                return
            raise

    scheduler_module.stop_scheduler = patched_stop_scheduler

    try:
        yield app
    finally:
        # Restore original methods
        scheduler_module.stop_scheduler = original_stop_scheduler
        # Reset scheduler instance between tests
        scheduler_module.scheduler_instance = None

        # Cleanup
        from app.database.connection import disconnect_from_db

        # Only disconnect, database cleanup is handled by fixture scope
        await disconnect_from_db()

        app.dependency_overrides.clear()
        # Restore original verify_auth_token function
        security_module.verify_auth_token = original_verify_auth_token


@pytest.fixture(scope="function")
def test_client(test_app):
    """Function-scoped fixture to create a TestClient for the test app.

    Note: test_app is async but yields a sync FastAPI instance.
    pytest-asyncio automatically handles async fixture dependencies for sync fixtures.
    TestClient itself is sync and works with the sync FastAPI instance yielded by test_app.
    """
    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture(scope="function")
def auth_token():
    """Function-scoped fixture providing a valid auth token."""
    return TEST_AUTH_TOKEN


@pytest.fixture(scope="function")
def auth_header(auth_token):
    """Function-scoped fixture providing Authorization header."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture(scope="function")
def goog_api_key_header():
    """Function-scoped fixture providing x-goog-api-key header."""
    return {"x-goog-api-key": TEST_AUTH_TOKEN}


@pytest.fixture(scope="function")
def auth_cookies(auth_token):
    """Function-scoped fixture providing auth_token cookie for cookie-based authentication."""
    return {"auth_token": auth_token}
