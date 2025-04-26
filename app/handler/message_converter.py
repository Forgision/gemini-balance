
from abc import ABC, abstractmethod
import json
import re
from typing import Any, Dict, List, Optional
import requests
import base64
import logging # Add logging

# Import settings and mappings
from app.config.config import settings, AUDIO_FORMAT_TO_MIMETYPE, VIDEO_FORMAT_TO_MIMETYPE
from app.core.constants import DATA_URL_PATTERN, IMAGE_URL_PATTERN, SUPPORTED_ROLES

logger = logging.getLogger(__name__) # Add a logger


class MessageConverter(ABC):
    """消息转换器基类"""

    @abstractmethod
    def convert(self, messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        pass

def _get_mime_type_and_data(base64_string):
    """
    从 base64 字符串中提取 MIME 类型和数据。
    
    参数:
        base64_string (str): 可能包含 MIME 类型信息的 base64 字符串
        
    返回:
        tuple: (mime_type, encoded_data)
    """
    # 检查字符串是否以 "data:" 格式开始
    if base64_string.startswith('data:'):
        # 提取 MIME 类型和数据
        pattern = DATA_URL_PATTERN
        match = re.match(pattern, base64_string)
        if match:
            mime_type = "image/jpeg" if match.group(1) == "image/jpg" else match.group(1)
            encoded_data = match.group(2)
            return mime_type, encoded_data
    
    # 如果不是预期格式，假定它只是数据部分
    return None, base64_string

def _convert_image(image_url: str) -> Dict[str, Any]:
    if image_url.startswith("data:image"):
        mime_type, encoded_data = _get_mime_type_and_data(image_url)
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": encoded_data
            }
        }
    else:
        encoded_data = _convert_image_to_base64(image_url)
        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": encoded_data
            }
        }


def _convert_image_to_base64(url: str) -> str:
    """
    将图片URL转换为base64编码
    Args:
        url: 图片URL
    Returns:
        str: base64编码的图片数据
    """
    response = requests.get(url)
    if response.status_code == 200:
        # 将图片内容转换为base64
        img_data = base64.b64encode(response.content).decode('utf-8')
        return img_data
    else:
        raise Exception(f"Failed to fetch image: {response.status_code}")


def _process_text_with_image(text: str) -> List[Dict[str, Any]]:
    """
    处理可能包含图片URL的文本，提取图片并转换为base64

    Args:
        text: 可能包含图片URL的文本

    Returns:
        List[Dict[str, Any]]: 包含文本和图片的部分列表
    """
    parts = []
    img_url_match = re.search(IMAGE_URL_PATTERN, text)
    if img_url_match:
        # 提取URL
        img_url = img_url_match.group(2)
        # 将URL对应的图片转换为base64
        try:
            base64_data = _convert_image_to_base64(img_url)
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64_data
                }
            })
        except Exception:
            # 如果转换失败，回退到文本模式
            parts.append({"text": text})
    else:
        # 没有图片URL，作为纯文本处理
        parts.append({"text": text})
    return parts


class OpenAIMessageConverter(MessageConverter):
    """OpenAI消息格式转换器"""

    def _validate_media_data(self, format: str, data: str, supported_formats: List[str], max_size: int) -> tuple[Optional[str], Optional[str]]:
        """Validates format and size of Base64 media data."""
        if format.lower() not in supported_formats:
            logger.error(f"Unsupported media format: {format}. Supported: {supported_formats}")
            raise ValueError(f"Unsupported media format: {format}")

        try:
            # Decode Base64 to check size
            # Be careful with memory usage for very large files
            # Consider streaming decoding or checking length heuristic first if memory is a concern
            decoded_data = base64.b64decode(data, validate=True) # Use validate=True for stricter check
            if len(decoded_data) > max_size:
                logger.error(f"Media data size ({len(decoded_data)} bytes) exceeds limit ({max_size} bytes).")
                raise ValueError(f"Media data size exceeds limit of {max_size // 1024 // 1024}MB")
            # No need to return decoded_data, just the original base64 if valid
            return data
        except base64.binascii.Error as e:
            logger.error(f"Invalid Base64 data provided: {e}")
            raise ValueError("Invalid Base64 data")
        except Exception as e:
            logger.error(f"Error validating media data: {e}")
            raise # Re-raise other potential errors

    def convert(self, messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        converted_messages = []
        system_instruction_parts = []

        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            parts = []

            # --- Start Modification ---
            if "content" in msg and isinstance(msg["content"], list):
                for content_item in msg["content"]:
                    if not isinstance(content_item, dict):
                        # Skip non-dict items if any unexpected format appears
                        logger.warning(f"Skipping unexpected content item format: {type(content_item)}")
                        continue

                    content_type = content_item.get("type")

                    if content_type == "text" and content_item.get("text"):
                        parts.append({"text": content_item["text"]})
                    elif content_type == "image_url" and content_item.get("image_url", {}).get("url"):
                        try:
                           parts.append(_convert_image(content_item["image_url"]["url"]))
                        except Exception as e:
                           logger.error(f"Failed to convert image URL {content_item['image_url']['url']}: {e}")
                           # Decide how to handle: skip part, add error text, etc.
                           parts.append({"text": f"[Error processing image: {content_item['image_url']['url']}]"})
                    # --- Add handling for input_audio ---
                    elif content_type == "input_audio" and content_item.get("input_audio"):
                        audio_info = content_item["input_audio"]
                        audio_data = audio_info.get("data")
                        audio_format = audio_info.get("format", "").lower()

                        if not audio_data or not audio_format:
                            logger.warning("Skipping audio part due to missing data or format.")
                            continue

                        try:
                            # Validate size and format
                            validated_data = self._validate_media_data(
                                audio_format,
                                audio_data,
                                settings.SUPPORTED_AUDIO_FORMATS,
                                settings.MAX_AUDIO_SIZE_BYTES
                            )

                            # Get MIME type
                            mime_type = AUDIO_FORMAT_TO_MIMETYPE.get(audio_format)
                            if not mime_type:
                                # Should not happen if format validation passed, but double-check
                                logger.error(f"Could not find MIME type for supported format: {audio_format}")
                                raise ValueError(f"Internal error: MIME type mapping missing for {audio_format}")

                            parts.append({
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": validated_data  # Use the validated Base64 data
                                }
                            })
                            logger.debug(f"Successfully added audio part (format: {audio_format})")

                        except ValueError as e:
                            logger.error(f"Skipping audio part due to validation error: {e}")
                            # Add placeholder text indicating the error
                            parts.append({"text": f"[Error processing audio: {e}]"})
                        except Exception as e:
                            logger.exception(f"Unexpected error processing audio part.")
                            parts.append({"text": "[Unexpected error processing audio]"})

                    # --- Add handling for input_video (similar pattern) ---
                    elif content_type == "input_video" and content_item.get("input_video"):
                        video_info = content_item["input_video"]
                        video_data = video_info.get("data")
                        video_format = video_info.get("format", "").lower()

                        if not video_data or not video_format:
                            logger.warning("Skipping video part due to missing data or format.")
                            continue

                        try:
                            validated_data = self._validate_media_data(
                                video_format,
                                video_data,
                                settings.SUPPORTED_VIDEO_FORMATS,
                                settings.MAX_VIDEO_SIZE_BYTES
                            )
                            mime_type = VIDEO_FORMAT_TO_MIMETYPE.get(video_format)
                            if not mime_type:
                                raise ValueError(f"Internal error: MIME type mapping missing for {video_format}")

                            parts.append({
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": validated_data
                                }
                            })
                            logger.debug(f"Successfully added video part (format: {video_format})")

                        except ValueError as e:
                             logger.error(f"Skipping video part due to validation error: {e}")
                             parts.append({"text": f"[Error processing video: {e}]"})
                        except Exception as e:
                             logger.exception(f"Unexpected error processing video part.")
                             parts.append({"text": "[Unexpected error processing video]"})
                    # --- End new media handling ---
                    else:
                        # Log unrecognized but present types
                        if content_type:
                            logger.warning(f"Unsupported content type or missing data in structured content: {content_type}")
                        # Silently ignore items without a 'type' or if structure is unexpected

            # --- End Modification for list content ---
            # Keep processing for simple string content (might contain image markdown)
            elif "content" in msg and isinstance(msg["content"], str) and msg["content"]:
                # This path handles simple text or markdown images.
                # If you expect audio/video ONLY via the structured list format,
                # this part remains as is. If you might have URLs in plain text,
                # you'd need more complex regex parsing here.
                parts.extend(_process_text_with_image(msg["content"]))
            elif "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                 # Keep existing tool call processing
                 for tool_call in msg["tool_calls"]:
                     function_call = tool_call.get("function",{})
                     # Sanitize arguments loading
                     arguments_str = function_call.get("arguments","{}")
                     try:
                         function_call["args"] = json.loads(arguments_str)
                     except json.JSONDecodeError:
                         logger.warning(f"Failed to decode tool call arguments: {arguments_str}")
                         function_call["args"] = {} # Assign empty dict on error
                     if "arguments" in function_call: # Check before deleting
                         # Ensure 'arguments' key exists before attempting deletion
                         # In some OpenAI versions, it might already be absent
                         pass # No explicit delete needed if structure is {'function': {'name': '...', 'args': ...}}
                     else:
                         # If 'arguments' was the source key, delete it after parsing
                         if 'arguments' in function_call: # Check again just in case
                            del function_call["arguments"]

                     parts.append({"functionCall": function_call})

            # Role assignment and message appending logic (keep as is)
            if role not in SUPPORTED_ROLES:
                if role == "tool":
                    role = "user" # Gemini uses 'user' role for function/tool responses
                # ... (rest of role handling logic) ...
                else:
                   # Fallback role logic
                   if idx == len(messages) - 1:
                       role = "user"
                   else:
                       # Previous logic assigned 'model'. Check if this is always correct.
                       # Tool/Function responses are usually 'model' in Gemini after the 'user' (tool result) turn.
                       role = "model" # Stick to 'model' as the default fallback for non-user/system/tool

            if parts:
                 if role == "system":
                     # Check if system instructions can contain media - unlikely based on Gemini docs
                     # Filter out non-text parts for safety?
                     text_only_parts = [p for p in parts if "text" in p]
                     if len(text_only_parts) != len(parts):
                         logger.warning("Non-text parts found in system message; discarding them.")
                     if text_only_parts:
                        system_instruction_parts.extend(text_only_parts)

                 else:
                     # Ensure role is mapped correctly ('model' for assistant turns, 'user' for tool result turns)
                     gemini_role = "model" if role == "assistant" else role # 'tool' role already mapped to 'user'
                     converted_messages.append({"role": gemini_role, "parts": parts})

        system_instruction = (
            None
            if not system_instruction_parts
            else {
                "role": "system", # Gemini supports a dedicated system instruction
                "parts": system_instruction_parts,
            }
        )
        # Gemini expects 'model' for assistant turns, and 'user' for function/tool responses.
        # The role mapping logic above should handle this correctly now.

        # Debug: Log the final converted structure before returning
        # logger.debug(f"Converted messages for Gemini: {json.dumps(converted_messages, indent=2)}")
        # if system_instruction:
        #     logger.debug(f"System instruction for Gemini: {json.dumps(system_instruction, indent=2)}")

        return converted_messages, system_instruction
