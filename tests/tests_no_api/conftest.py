"""
Configuration file for tests_no_api integration tests.
Provides fixtures for in-memory databases, mocked API clients, and test application setup.
"""

import pytest
import pytest_asyncio
import asyncio
import importlib
import json
import datetime
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator, Dict, Any, List

from pytest import MonkeyPatch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from sqlalchemy import create_engine, select, insert, update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from app.config.config import settings
from app.core.application import create_app
from app.service.key.key_manager import KeyManager, AsyncSessionLocal as KeyManagerAsyncSessionLocal
from app.dependencies import get_key_manager


# Test constants
TEST_API_KEYS = ["test_key_1", "test_key_2", "test_key_3"]
TEST_VERTEX_API_KEYS = ["test_vertex_key_1", "test_vertex_key_2"]
TEST_AUTH_TOKEN = "test_auth_token_12345"
TEST_ALLOWED_TOKENS = ["test_token_1", "test_token_2"]


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
    monkeypatch_session.setattr(settings, "SQLITE_DATABASE", ":memory:")
    
    # Patch KeyManager database
    monkeypatch_session.setattr(settings, "KEY_MATRIX_DB_URL", "sqlite+aiosqlite:///:memory:")
    
    # Patch test configuration
    monkeypatch_session.setattr(settings, "API_KEYS", TEST_API_KEYS)
    monkeypatch_session.setattr(settings, "VERTEX_API_KEYS", TEST_VERTEX_API_KEYS)
    monkeypatch_session.setattr(settings, "AUTH_TOKEN", TEST_AUTH_TOKEN)
    monkeypatch_session.setattr(settings, "ALLOWED_TOKENS", TEST_ALLOWED_TOKENS)
    
    import app.database.connection as db_conn
    importlib.reload(db_conn)

    from databases import Database
    db_conn.DATABASE_URL = "sqlite:///:memory:"
    db_conn.database = Database(db_conn.DATABASE_URL)
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    db_conn.engine = create_engine(
        db_conn.DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    
    # Reload KeyManager module to apply new DB URL
    import app.service.key.key_manager as km
    importlib.reload(km)


@pytest.fixture(scope="session")
def in_memory_db_engine(patch_database_settings):
    """Session-scoped fixture to create in-memory SQLite engine for main database."""
    from app.database.connection import Base
    
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose(close=True)


@pytest.fixture(scope="session")
def key_manager_async_engine(patch_database_settings):
    """Session-scoped fixture to create in-memory async SQLite engine for KeyManager."""
    from app.service.key.key_manager import Base
    
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    
    # Create tables
    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    asyncio.run(create_tables())
    
    yield engine
    
    # Cleanup
    async def dispose_engine():
        await engine.dispose(close=True)
    
    asyncio.run(dispose_engine())


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
                        "role": "model"
                    },
                    "finishReason": "STOP"
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
        return {
            "embeddings": [
                {"values": [0.1] * 768},
                {"values": [0.2] * 768}
            ]
        }
    
    async def get_models_side_effect(api_key):
        """Mock get_models method."""
        return {
            "models": [
                {
                    "name": "models/gemini-pro",
                    "displayName": "Gemini Pro",
                    "description": "Best model for general tasks",
                    "supportedGenerationMethods": ["generateContent", "streamGenerateContent"]
                },
                {
                    "name": "models/gemini-2.0-flash-exp",
                    "displayName": "Gemini 2.0 Flash Experimental",
                    "description": "Fast experimental model",
                    "supportedGenerationMethods": ["generateContent", "streamGenerateContent"]
                }
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
                        "content": "This is a mock response from OpenAI API."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    
    async def stream_generate_content_side_effect(payload, model, api_key):
        """Mock stream_generate_content method - returns OpenAI SSE chunks."""
        chunks = [
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"' + model + '","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"' + model + '","choices":[{"index":0,"delta":{"content":" "},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"' + model + '","choices":[{"index":0,"delta":{"content":"world"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"' + model + '","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n'
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
                    "owned_by": "google"
                },
                {
                    "id": "gemini-2.0-flash-exp",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "google"
                }
            ]
        }
    
    async def create_embeddings_side_effect(input, model, api_key):
        """Mock create_embeddings method."""
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1] * 768,
                    "index": 0
                }
            ],
            "model": model,
            "usage": {
                "prompt_tokens": 1,
                "total_tokens": 1
            }
        }
    
    async def generate_images_side_effect(payload, api_key):
        """Mock generate_images method."""
        return {
            "created": 1234567890,
            "data": [
                {
                    "url": "https://example.com/image.png",
                    "revised_prompt": payload.get("prompt", "")
                }
            ]
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
            from openai.types.embeddings import Embedding
            try:
                from openai.types.embeddings import Usage
            except ImportError:
                from openai.types import Usage
            
            num_items = len(input) if isinstance(input, list) else 1
            embedding_list = [
                Embedding(object="embedding", index=i, embedding=[0.1] * 768)
                for i in range(num_items)
            ]
            usage_obj = Usage(prompt_tokens=10, total_tokens=10)
            return CreateEmbeddingResponse(
                object="list",
                data=embedding_list,
                model=model,
                usage=usage_obj
            )
        except (ImportError, AttributeError, TypeError) as e:
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
                EmbeddingData(index=i, embedding=[0.1] * 768)
                for i in range(num_items)
            ]
            usage_obj = UsageData()
            return CreateEmbeddingResponseModel(
                object="list",
                data=[e.model_dump() for e in embedding_list],
                model=model,
                usage=usage_obj
            )
    
    mock_openai_client.embeddings.create = MagicMock(side_effect=create_embedding_side_effect)
    
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
    
    mock_genai_client.models.generate_images = MagicMock(side_effect=generate_images_side_effect)
    
    # Patch OpenAI and Gemini clients
    monkeypatch.setattr(openai, "OpenAI", lambda *args, **kwargs: mock_openai_client)
    monkeypatch.setattr(genai, "Client", lambda *args, **kwargs: mock_genai_client)
    
    return {
        'openai_client': mock_openai_client,
        'genai_client': mock_genai_client
    }


@pytest.fixture(scope="function")
def patched_api_clients(mock_gemini_api_client, mock_openai_api_client):
    """Function-scoped fixture to patch API clients in the application."""
    # Create factory functions that return our mocks when called
    def gemini_factory(*args, **kwargs):
        return mock_gemini_api_client
    
    def openai_factory(*args, **kwargs):
        return mock_openai_api_client
    
    # Patch the classes so that when services instantiate them, they get our mocks
    with patch('app.service.client.api_client.GeminiApiClient', new=gemini_factory), \
         patch('app.service.client.api_client.OpenaiApiClient', new=openai_factory), \
         patch('app.service.chat.gemini_chat_service.GeminiApiClient', new=gemini_factory), \
         patch('app.service.chat.openai_chat_service.GeminiApiClient', new=gemini_factory), \
         patch('app.service.chat.vertex_express_chat_service.GeminiApiClient', new=gemini_factory), \
         patch('app.service.openai_compatiable.openai_compatiable_service.OpenaiApiClient', new=openai_factory), \
         patch('app.service.embedding.gemini_embedding_service.GeminiApiClient', new=gemini_factory), \
         patch('app.service.files.files_service.GeminiApiClient', new=gemini_factory), \
         patch('app.service.model.model_service.GeminiApiClient', new=gemini_factory), \
         patch('app.service.claude_proxy_service.GeminiApiClient', new=gemini_factory):
        yield {
            'gemini': mock_gemini_api_client,
            'openai': mock_openai_api_client
        }


@pytest_asyncio.fixture(scope="function")
async def test_key_manager(key_manager_async_engine):
    """Function-scoped fixture to create and initialize a real KeyManager instance."""
    # Create new async session maker for this test
    async_session_maker = async_sessionmaker(
        key_manager_async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False
    )
    
    # Patch init_database to use our test engine
    from app.service.key import key_manager as km_module
    
    async def patched_init_database():
        """Patched init_database that uses the test engine."""
        from app.service.key.key_manager import Base
        async with key_manager_async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    # Temporarily patch the init_database function
    original_init_database = km_module.init_database
    km_module.init_database = patched_init_database
    
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
            async_session_maker=async_session_maker
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
        except Exception as e:
            # If init fails, try with default
            await key_manager.init()
        
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
        km_module.init_database = original_init_database
        asyncio_module.create_task = original_create_task


@pytest.fixture(scope="function")
def test_app(in_memory_db_engine, test_key_manager, patched_api_clients, patched_service_clients):
    """Function-scoped fixture to create a test FastAPI app using the real create_app() function."""
    # Ensure KeyManager is initialized (it should be from test_key_manager fixture, but verify)
    async def ensure_key_manager_init():
        if not test_key_manager.is_ready:
            await test_key_manager.init()
    
    asyncio.run(ensure_key_manager_init())
    
    # FULLY initialize database with required config BEFORE create_app() runs
    from app.database.connection import database
    from app.database.initialization import initialize_database
    from app.config.config import sync_initial_settings
    from app.database.models import Settings as SettingsModel
    
    # Step 1: Initialize database (create tables and import settings)
    initialize_database()
    
    # Step 2: Connect to database and ensure test auth tokens are in database
    async def setup_db_and_config():
        if not database.is_connected:
            await database.connect()
        
        # Step 3: Sync initial settings first (this creates t_settings table and syncs config)
        await sync_initial_settings()
        
        # Step 4: Ensure TEST_AUTH_TOKEN and TEST_ALLOWED_TOKENS are in database AFTER sync
        # This ensures our test tokens override any defaults from sync
        try:
            # Check if AUTH_TOKEN exists in database
            query = select(SettingsModel.key, SettingsModel.value).where(
                SettingsModel.key.in_(["AUTH_TOKEN", "ALLOWED_TOKENS"])
            )
            existing = {row["key"]: row["value"] for row in await database.fetch_all(query)}
            
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
                await database.execute(query_insert)
            elif existing["AUTH_TOKEN"] != TEST_AUTH_TOKEN:
                query_update = (
                    update(SettingsModel)
                    .where(SettingsModel.key == "AUTH_TOKEN")
                    .values(value=TEST_AUTH_TOKEN, updated_at=now)
                )
                await database.execute(query_update)
            
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
                await database.execute(query_insert)
            elif existing["ALLOWED_TOKENS"] != allowed_tokens_json:
                query_update = (
                    update(SettingsModel)
                    .where(SettingsModel.key == "ALLOWED_TOKENS")
                    .values(value=allowed_tokens_json, updated_at=now)
                )
                await database.execute(query_update)
            
            # Re-sync to load our test tokens into memory settings
            await sync_initial_settings()
            
        except Exception as e:
            # If database setup fails, ensure settings are still patched
            pass
        
        # Step 5: Ensure settings object ALWAYS has test tokens (final override)
        settings.AUTH_TOKEN = TEST_AUTH_TOKEN
        settings.ALLOWED_TOKENS = TEST_ALLOWED_TOKENS
    
    asyncio.run(setup_db_and_config())
    
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
    from app.core.security import SecurityService, verify_auth_token
    from app.router import gemini_routes, openai_routes, openai_compatible_routes, key_routes, claude_routes
    from app.router import config_routes, error_log_routes, scheduler_routes, stats_routes
    
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
    
    security_service = SecurityService()
    
    async def mock_verify_key_or_goog_api_key(key=None, x_goog_api_key=None):
        """Mock verify_key_or_goog_api_key to accept TEST_AUTH_TOKEN and TEST_ALLOWED_TOKENS."""
        # If key in URL is provided and valid, return it
        if key is not None and (key == TEST_AUTH_TOKEN or key in TEST_ALLOWED_TOKENS):
            return key
        # Otherwise check x-goog-api-key header
        if x_goog_api_key is not None:
            if x_goog_api_key == TEST_AUTH_TOKEN or x_goog_api_key in TEST_ALLOWED_TOKENS:
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
        auth_token = request.cookies.get("auth_token") if hasattr(request, 'cookies') else None
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
    
    app.dependency_overrides[
        openai_routes.security_service.verify_authorization
    ] = mock_verify_authorization
    
    app.dependency_overrides[
        openai_compatible_routes.security_service.verify_authorization
    ] = mock_verify_authorization
    
    app.dependency_overrides[
        claude_routes.security_service.verify_auth_token
    ] = mock_verify_auth_token_header
    
    # Override verify_token dependencies for scheduler and stats routes
    if hasattr(scheduler_routes, 'verify_token'):
        app.dependency_overrides[
            scheduler_routes.verify_token
        ] = mock_verify_token_for_dependencies
    
    if hasattr(stats_routes, 'verify_token'):
        app.dependency_overrides[
            stats_routes.verify_token
        ] = mock_verify_token_for_dependencies
    
    # Ensure KeyManager is stored in app.state (should already be set, but ensure it)
    app.state.key_manager = test_key_manager
    
    # Database is already initialized and connected, settings are already synced
    # Just ensure app.state is properly set up
    if not hasattr(app.state, "key_manager"):
        app.state.key_manager = test_key_manager
    
    # Patch ConfigService to prevent KeyManager recreation during update_config/reset_config
    # This prevents event loop issues when KeyManager background tasks are involved
    from app.service.config import config_service as config_service_module
    original_update_config = config_service_module.ConfigService.update_config
    original_reset_config = config_service_module.ConfigService.reset_config
    
    @staticmethod
    async def patched_update_config(config_data: Dict[str, Any], app: FastAPI) -> Dict[str, Any]:
        """Patched update_config that skips KeyManager recreation in tests."""
        # Call original method but skip the KeyManager recreation part
        # We'll manually handle the settings update and keep the test KeyManager
        
        # First, update settings in memory (this is done in original_update_config)
        for key, value in config_data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        
        # Get existing settings and update database (same logic as original)
        from app.database.services import get_all_settings
        existing_settings_raw: List[Dict[str, Any]] = await get_all_settings()
        existing_settings_map: Dict[str, Dict[str, Any]] = {
            s["key"]: s for s in existing_settings_raw
        }
        existing_keys = set(existing_settings_map.keys())
        
        settings_to_update: List[Dict[str, Any]] = []
        settings_to_insert: List[Dict[str, Any]] = []
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
        
        for key, value in config_data.items():
            if isinstance(value, list):
                db_value = json.dumps(value)
            elif isinstance(value, dict):
                db_value = json.dumps(value)
            elif isinstance(value, bool):
                db_value = str(value).lower()
            else:
                db_value = str(value)
            
            if key in existing_keys and existing_settings_map[key]["value"] == db_value:
                continue
            
            data = {
                "key": key,
                "value": db_value,
                "description": f"{key} configuration item",
                "updated_at": now,
            }
            
            if key in existing_keys:
                data["description"] = existing_settings_map[key].get("description", data["description"])
                settings_to_update.append(data)
            else:
                data["created_at"] = now
                settings_to_insert.append(data)
        
        # Update database
        from app.database.connection import database
        from app.database.models import Settings as SettingsModel
        if settings_to_insert or settings_to_update:
            async with database.transaction():
                if settings_to_insert:
                    query_insert = insert(SettingsModel).values(settings_to_insert)
                    await database.execute(query=query_insert)
                if settings_to_update:
                    for setting_data in settings_to_update:
                        query_update = (
                            update(SettingsModel)
                            .where(SettingsModel.key == setting_data["key"])
                            .values(
                                value=setting_data["value"],
                                description=setting_data["description"],
                                updated_at=setting_data["updated_at"],
                            )
                        )
                        await database.execute(query=query_update)
        
        # Skip KeyManager recreation - keep using test KeyManager
        # In a real app, KeyManager would be recreated, but in tests we keep the test instance
        
        return await config_service_module.ConfigService.get_config()
    
    @staticmethod
    async def patched_reset_config(app: FastAPI) -> Dict[str, Any]:
        """Patched reset_config that skips KeyManager recreation in tests."""
        # Reload settings from env/dotenv
        from dotenv import find_dotenv, load_dotenv
        from app.config.config import Settings as ConfigSettings
        load_dotenv(find_dotenv(), override=True)
        for key, value in ConfigSettings().model_dump().items():
            setattr(settings, key, value)
        
        # Sync settings to database
        from app.config.config import sync_initial_settings
        await sync_initial_settings()
        
        # Skip KeyManager recreation - keep using test KeyManager
        # The test KeyManager already has the correct keys from TEST_API_KEYS
        
        return await config_service_module.ConfigService.get_config()
    
    # Apply patches
    config_service_module.ConfigService.update_config = patched_update_config
    config_service_module.ConfigService.reset_config = patched_reset_config
    
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
                scheduler_logger.info("Scheduler event loop is closed (likely already stopped).")
                # Reset scheduler_instance to None to allow restart
                scheduler_module.scheduler_instance = None
                return
            raise
    
    scheduler_module.stop_scheduler = patched_stop_scheduler
    
    try:
        yield app
    finally:
        # Restore original methods
        config_service_module.ConfigService.update_config = original_update_config
        config_service_module.ConfigService.reset_config = original_reset_config
        scheduler_module.stop_scheduler = original_stop_scheduler
        # Reset scheduler instance between tests
        scheduler_module.scheduler_instance = None
    
    # Cleanup
    async def cleanup():
        from app.database.connection import disconnect_from_db
        # Only disconnect, database cleanup is handled by fixture scope
        if database.is_connected:
            await disconnect_from_db()
    
    asyncio.run(cleanup())
    app.dependency_overrides.clear()
    # Restore original verify_auth_token function
    security_module.verify_auth_token = original_verify_auth_token


@pytest.fixture(scope="function")
def test_client(test_app):
    """Function-scoped fixture to create a TestClient for the test app."""
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

