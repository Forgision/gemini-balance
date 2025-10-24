"""
Native Gemini TTS extension data models
Inherits from original models, adding native Gemini TTS-related fields while maintaining backward compatibility
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from app.domain.gemini_models import GenerationConfig as BaseGenerationConfig


class TTSGenerationConfig(BaseGenerationConfig):
    """
    Generation configuration class with TTS support
    Inherits from the original GenerationConfig, adding TTS-related fields
    """

    # TTS-related configuration
    responseModalities: Optional[List[str]] = None
    speechConfig: Optional[Dict[str, Any]] = None


class MultiSpeakerVoiceConfig(BaseModel):
    """Multi-speaker voice configuration"""

    speakerVoiceConfigs: List[Dict[str, Any]]


class SpeechConfig(BaseModel):
    """Speech configuration"""

    multiSpeakerVoiceConfig: Optional[MultiSpeakerVoiceConfig] = None
    voiceConfig: Optional[Dict[str, Any]] = None


class TTSRequest(BaseModel):
    """TTS request model"""

    contents: List[Dict[str, Any]]
    generationConfig: TTSGenerationConfig
