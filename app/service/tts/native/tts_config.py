"""
TTS Extension Configuration
Controls whether to enable TTS functionality
"""

import os
from typing import Union
from app.service.chat.gemini_chat_service import GeminiChatService
from app.service.tts.native.tts_chat_service import TTSGeminiChatService


class TTSConfig:
    """TTS Configuration Management"""
    
    @staticmethod
    def is_tts_enabled() -> bool:
        """
        Check if TTS functionality is enabled
        Controlled by the environment variable ENABLE_TTS, defaults to False
        """
        return os.getenv("ENABLE_TTS", "false").lower() in ("true", "1", "yes", "on")
    
    @staticmethod
    def get_chat_service(base_url: str, key_manager) -> Union[GeminiChatService, TTSGeminiChatService]:
        """
        Factory method: returns the appropriate chat service based on the configuration
        """
        if TTSConfig.is_tts_enabled():
            return TTSGeminiChatService(base_url, key_manager)
        else:
            return GeminiChatService(base_url, key_manager)


# Convenience function
def create_chat_service(base_url: str, key_manager) -> Union[GeminiChatService, TTSGeminiChatService]:
    """Create a chat service instance"""
    return TTSConfig.get_chat_service(base_url, key_manager)
