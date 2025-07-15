"""
TTS路由扩展
提供多人TTS增强服务
"""

from fastapi import Depends

from app.config.config import settings
from app.service.key.key_manager import KeyManager, get_key_manager_instance
from app.service.tts.multi_speaker.tts_chat_service import TTSGeminiChatService


async def get_key_manager():
    """获取密钥管理器实例"""
    return get_key_manager_instance()


async def get_tts_chat_service(key_manager: KeyManager = Depends(get_key_manager)) -> TTSGeminiChatService:
    """
    获取多人TTS增强聊天服务实例
    """
    return TTSGeminiChatService(settings.BASE_URL, key_manager)


