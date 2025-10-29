from fastapi import Depends
from app.config.config import settings
from app.service.chat.openai_chat_service import OpenAIChatService
from app.service.key.key_manager import KeyManager, get_key_manager_instance

async def get_key_manager():
    return await get_key_manager_instance(settings.API_KEYS, settings.VERTEX_API_KEYS)

from app.service.chat.gemini_chat_service import GeminiChatService

def get_openai_chat_service(
    key_manager: KeyManager = Depends(get_key_manager),
) -> OpenAIChatService:
    """Get the OpenAI chat service instance."""
    return OpenAIChatService(settings.BASE_URL, key_manager)

from app.service.chat.vertex_express_chat_service import (
    GeminiChatService as VertexGeminiChatService,
)

def get_gemini_chat_service(
    key_manager: KeyManager = Depends(get_key_manager),
) -> GeminiChatService:
    """Get the Gemini chat service instance."""
    return GeminiChatService(settings.BASE_URL, key_manager)

from app.service.openai_compatiable.openai_compatiable_service import (
    OpenAICompatiableService,
)

def get_vertex_express_chat_service(
    key_manager: KeyManager = Depends(get_key_manager),
) -> VertexGeminiChatService:
    """Get the Vertex Express chat service instance."""
    return VertexGeminiChatService(settings.VERTEX_EXPRESS_BASE_URL, key_manager)

from app.service.config.config_service import ConfigService

def get_openai_compatible_chat_service(
    key_manager: KeyManager = Depends(get_key_manager),
) -> OpenAICompatiableService:
    """Get the OpenAI compatible chat service instance."""
    return OpenAICompatiableService(settings.BASE_URL, key_manager)

from app.service.error_log import error_log_service

def get_config_service() -> ConfigService:
    """Get the config service instance."""
    return ConfigService()

from app.service.files.files_service import FilesService

from typing import Any

def get_error_log_service() -> Any:
    """Get the error log service instance."""
    return error_log_service

async def get_files_service() -> FilesService:
    """Get the files service instance."""
    return FilesService()
