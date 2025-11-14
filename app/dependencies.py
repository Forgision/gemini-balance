from typing import Any

from fastapi import Depends, Request

from app.config.config import settings
from app.service.chat.gemini_chat_service import GeminiChatService
from app.service.chat.openai_chat_service import OpenAIChatService
from app.service.chat.vertex_express_chat_service import (
    GeminiChatService as VertexGeminiChatService,
)
from app.service.config.config_service import ConfigService
from app.service.error_log import error_log_service
from app.service.files.files_service import FilesService
from app.service.key.key_manager import KeyManager
from app.service.openai_compatiable.openai_compatiable_service import (
    OpenAICompatiableService,
)


async def get_key_manager(request: Request) -> KeyManager:
    """Get KeyManager instance from app.state."""
    if not hasattr(request.app.state, "key_manager"):
        raise RuntimeError("KeyManager not initialized. Check application startup.")
    return request.app.state.key_manager


def get_openai_chat_service(
    key_manager: KeyManager = Depends(get_key_manager),
) -> OpenAIChatService:
    """Get the OpenAI chat service instance."""
    return OpenAIChatService(settings.BASE_URL, key_manager)


def get_gemini_chat_service(
    key_manager: KeyManager = Depends(get_key_manager),
) -> GeminiChatService:
    """Get the Gemini chat service instance."""
    return GeminiChatService(settings.BASE_URL, key_manager)


def get_vertex_express_chat_service(
    key_manager: KeyManager = Depends(get_key_manager),
) -> VertexGeminiChatService:
    """Get the Vertex Express chat service instance."""
    return VertexGeminiChatService(settings.VERTEX_EXPRESS_BASE_URL, key_manager)


def get_openai_compatible_chat_service(
    key_manager: KeyManager = Depends(get_key_manager),
) -> OpenAICompatiableService:
    """Get the OpenAI compatible chat service instance."""
    return OpenAICompatiableService(settings.BASE_URL, key_manager)


def get_config_service() -> ConfigService:
    """Get the config service instance."""
    return ConfigService()


def get_error_log_service() -> Any:
    """Get the error log service instance."""
    return error_log_service


async def get_files_service() -> FilesService:
    """Get the files service instance."""
    return FilesService()
