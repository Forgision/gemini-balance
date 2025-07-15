"""
多人对话TTS功能模块
Multi-speaker TTS functionality for conversation scenarios
"""

from .tts_chat_service import TTSGeminiChatService
from .tts_models import TTSGenerationConfig, MultiSpeakerVoiceConfig, SpeechConfig, TTSRequest
from .tts_response_handler import TTSResponseHandler
from .tts_routes import get_tts_chat_service

__all__ = [
    "TTSGeminiChatService",
    "TTSGenerationConfig",
    "MultiSpeakerVoiceConfig",
    "SpeechConfig",
    "TTSRequest",
    "TTSResponseHandler",
    "get_tts_chat_service"
]
