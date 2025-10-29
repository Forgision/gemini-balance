"""
TTS Route Extension
Provides native Gemini TTS enhanced service, supporting single and multi-speaker voice
"""

from fastapi import Depends

from app.config.config import settings
from app.service.key.key_manager import KeyManager
from app.dependencies import get_key_manager
from app.service.tts.native.tts_chat_service import TTSGeminiChatService


async def get_tts_chat_service(
    key_manager: KeyManager = Depends(get_key_manager),
) -> TTSGeminiChatService:
    """
    Get the native Gemini TTS enhanced chat service instance, supporting single and multi-speaker voice
    """
    return TTSGeminiChatService(settings.BASE_URL, key_manager)
