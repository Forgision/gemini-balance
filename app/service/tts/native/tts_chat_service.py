"""
Native Gemini TTS chat service extension
Inherits from the original chat service, adding native Gemini TTS support (single and multi-speaker), maintaining backward compatibility
"""

import datetime
import time
from typing import Any, Dict

from app.config.config import settings
from app.database.services import add_error_log, add_request_log
from app.domain.gemini_models import GeminiRequest
from app.log.logger import get_gemini_logger
from app.service.chat.gemini_chat_service import GeminiChatService
from app.service.tts.native.tts_response_handler import TTSResponseHandler

logger = get_gemini_logger()


class TTSGeminiChatService(GeminiChatService):
    """
    Gemini chat service with TTS support
    Inherits from the original GeminiChatService, adding TTS functionality
    """

    def __init__(self, base_url: str, key_manager):
        """
        Initialize the TTS chat service
        """
        super().__init__(base_url, key_manager)
        # Replace the original processor with the TTS response handler
        self.response_handler = TTSResponseHandler()
        logger.info(
            "TTS Gemini Chat Service initialized with multi-speaker TTS support"
        )

    async def generate_content(
        self, model: str, request: GeminiRequest, api_key: str
    ) -> Dict[str, Any]:
        """
        Generate content, with TTS support
        """
        try:
            # Add debug logs
            logger.info(f"TTS request model: {model}")
            logger.info(f"TTS request generationConfig: {request.generationConfig}")

            # Check if it's a TTS model, if so, special handling is required
            if "tts" in model.lower():
                logger.info("Detected TTS model, applying TTS-specific processing")
                # For TTS models, we need to ensure the correct fields are passed
                response = await self._handle_tts_request(model, request, api_key)
                return response
            else:
                # For non-TTS models, use the parent class's method
                response = await super().generate_content(model, request, api_key)
                return response
        except Exception as e:
            logger.error(f"TTS API call failed with error: {e}", exc_info=True)
            raise

    async def _handle_tts_request(
        self, model: str, request: GeminiRequest, api_key: str
    ) -> Dict[str, Any]:
        """
        Handle TTS-specific requests, including full logging functionality
        """
        # Record the start time and request time
        start_time = time.perf_counter()
        request_datetime = datetime.datetime.now()
        is_success = False
        status_code = None

        try:
            # Build a TTS-specific payload - without tools and safetySettings
            from app.service.chat.gemini_chat_service import _filter_empty_parts

            request_dict = request.model_dump(exclude_none=False)

            # Build a simplified payload for TTS
            payload = {
                "contents": _filter_empty_parts(request_dict.get("contents", [])),
                "generationConfig": request_dict.get("generationConfig", {}),
            }

            # Only add systemInstruction if it exists
            if request_dict.get("systemInstruction"):
                payload["systemInstruction"] = request_dict.get("systemInstruction")

            # Ensure generationConfig is not None
            if payload["generationConfig"] is None:
                payload["generationConfig"] = {}

            # Get TTS-related fields directly from request.generationConfig
            if request.generationConfig:
                # Add TTS-specific fields
                if request.generationConfig.responseModalities:
                    payload["generationConfig"]["responseModalities"] = (
                        request.generationConfig.responseModalities
                    )
                    logger.info(
                        f"Added responseModalities: {request.generationConfig.responseModalities}"
                    )

                if request.generationConfig.speechConfig:
                    payload["generationConfig"]["speechConfig"] = (
                        request.generationConfig.speechConfig
                    )
                    logger.info(
                        f"Added speechConfig: {request.generationConfig.speechConfig}"
                    )
            else:
                logger.warning(
                    "No generationConfig found in request, TTS fields may be missing"
                )

            logger.info(f"TTS payload before API call: {payload}")

            # Call the API
            response = await self.api_client.generate_content(payload, model, api_key)

            # If we get here, the API call was successful
            is_success = True
            status_code = 200

            # Use the TTS response handler to process the response
            return self.response_handler.handle_response(response, model, False, None)

        except Exception as e:
            # Log the error
            is_success = False
            error_msg = str(e)

            # Try to extract the status code from the error message
            import re

            match = re.search(r"status code (\d+)", error_msg)
            if match:
                status_code = int(match.group(1))
            else:
                status_code = 500

            # Add an error log
            await add_error_log(
                gemini_key=api_key,
                model_name=model,
                error_type="tts-api-error",
                error_log=error_msg,
                error_code=status_code,
                request_msg=(
                    request.model_dump(exclude_none=False)
                    if settings.ERROR_LOG_RECORD_REQUEST_BODY
                    else None
                ),
            )

            logger.error(f"TTS API call failed: {error_msg}", exc_info=True)
            raise

        finally:
            # Log the request
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)

            await add_request_log(
                model_name=model,
                api_key=api_key,
                is_success=is_success,
                status_code=status_code,
                latency_ms=latency_ms,
                request_time=request_datetime,
            )
