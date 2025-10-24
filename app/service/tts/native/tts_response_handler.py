"""
Native Gemini TTS response handler extension
Inherits from the original response handler, adding native Gemini TTS support, while maintaining backward compatibility
"""

from typing import Any, Dict, Optional
from app.handler.response_handler import GeminiResponseHandler
from app.log.logger import get_gemini_logger

logger = get_gemini_logger()


class TTSResponseHandler(GeminiResponseHandler):
    """
    Response handler with TTS support
    Inherits from the original GeminiResponseHandler, adding TTS response handling
    """

    def handle_response(
        self,
        response: Dict[str, Any],
        model: str,
        stream: bool = False,
        usage_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle the response, with support for TTS audio data
        """
        # Check if it's a TTS response (contains audio data)
        if self._is_tts_response(response):
            logger.info(
                "Detected TTS response with audio data, returning original response"
            )
            return response

        # For non-TTS responses, use the parent class's handling method
        return super().handle_response(response, model, stream, usage_metadata)

    def _is_tts_response(self, response: Dict[str, Any]) -> bool:
        """
        Check if it's a TTS response
        """
        try:
            if (
                response.get("candidates")
                and len(response["candidates"]) > 0
                and response["candidates"][0].get("content")
                and response["candidates"][0]["content"].get("parts")
                and len(response["candidates"][0]["content"]["parts"]) > 0
            ):
                parts = response["candidates"][0]["content"]["parts"]
                for part in parts:
                    if "inlineData" in part:
                        mime_type = part["inlineData"].get("mimeType", "")
                        if mime_type.startswith("audio/"):
                            return True
            return False
        except Exception as e:
            logger.warning(f"Error checking TTS response: {e}")
            return False
