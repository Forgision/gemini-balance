"""
TTS路由扩展
可选的路由覆盖，用于启用TTS功能
使用时可以替换原始路由的依赖注入
"""

from fastapi import Depends
from typing import Union

from app.config.config import settings
from app.service.key.key_manager import KeyManager, get_key_manager_instance
from app.service.chat.gemini_chat_service import GeminiChatService
from app.service.tts.multi_speaker.tts_chat_service import TTSGeminiChatService
from app.service.tts.multi_speaker.tts_config import TTSConfig


async def get_key_manager():
    """获取密钥管理器实例"""
    return get_key_manager_instance()


async def get_tts_chat_service(key_manager: KeyManager = Depends(get_key_manager)) -> Union[GeminiChatService, TTSGeminiChatService]:
    """
    获取聊天服务实例（支持TTS）
    根据配置返回原始服务或TTS增强服务
    """
    return TTSConfig.get_chat_service(settings.BASE_URL, key_manager)


# 使用说明：
# 要启用TTS功能，需要：
# 1. 设置环境变量 ENABLE_TTS=true
# 2. 在路由中使用 get_tts_chat_service 替换 get_chat_service
# 
# 例如在 gemini_routes.py 中：
# from app.service.tts.multi_speaker.tts_routes import get_tts_chat_service
# 
# async def generate_content(
#     chat_service = Depends(get_tts_chat_service)  # 替换原来的依赖
# ):
#     ...
