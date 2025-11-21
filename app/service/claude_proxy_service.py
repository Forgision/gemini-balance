"""
Claude Proxy Service
"""

import json
import uuid
import time
import datetime
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

import litellm
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.config import settings
from app.exception.api_exceptions import ApiClientException
from app.log.logger import get_gemini_logger
from app.service.client.api_client import GeminiApiClient
from app.database.services import add_error_log, add_request_log, update_usage_stats
from app.handler.response_handler import GeminiResponseHandler


logger = get_gemini_logger()


# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        schema.pop("default", None)
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(
                    f"Removing unsupported format '{schema['format']}' for string type in Gemini schema."
                )
                schema.pop("format")
        for key, value in list(schema.items()):
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        return [clean_gemini_schema(item) for item in schema]
    return schema


# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None

    @field_validator("model")
    def validate_model_field(cls, v, info):
        original_model = v
        new_model = v

        clean_v = v.split("/")[-1]

        mapped = False
        if settings.CLAUDE_PREFERRED_PROVIDER == "anthropic":
            new_model = f"anthropic/{clean_v}"
            mapped = True
        elif "haiku" in clean_v.lower():
            # Map to Gemini model name directly (no gemini/ prefix)
            new_model = settings.CLAUDE_SMALL_MODEL
            mapped = True
        elif "sonnet" in clean_v.lower():
            # Map to Gemini model name directly (no gemini/ prefix)
            new_model = settings.CLAUDE_BIG_MODEL
            mapped = True

        if mapped:
            logger.debug(f"Model mapping: '{original_model}' -> '{new_model}'")
        else:
            if not v.startswith(("openai/", "gemini/", "anthropic/")):
                logger.warning(
                    f"No prefix or mapping rule for model: '{original_model}'. Using as is."
                )
            new_model = v

        if hasattr(info, "data") and isinstance(info.data, dict):
            info.data["original_model"] = original_model

        return new_model


class TokenCountResponse(BaseModel):
    input_tokens: int


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None

    @field_validator("model")
    def validate_model_field(cls, v, info):
        original_model = v
        new_model = v

        clean_v = v.split("/")[-1]

        mapped = False
        if settings.CLAUDE_PREFERRED_PROVIDER == "anthropic":
            new_model = f"anthropic/{clean_v}"
            mapped = True
        elif "haiku" in clean_v.lower():
            # Map to Gemini model name directly (no gemini/ prefix)
            new_model = settings.CLAUDE_SMALL_MODEL
            mapped = True
        elif "sonnet" in clean_v.lower():
            # Map to Gemini model name directly (no gemini/ prefix)
            new_model = settings.CLAUDE_BIG_MODEL
            mapped = True

        if mapped:
            logger.debug(f"Model mapping: '{original_model}' -> '{new_model}'")
        else:
            if not v.startswith(("openai/", "gemini/", "anthropic/")):
                logger.warning(
                    f"No prefix or mapping rule for model: '{original_model}'. Using as is."
                )
            new_model = v

        if hasattr(info, "data") and isinstance(info.data, dict):
            info.data["original_model"] = original_model

        return new_model


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


class ClaudeProxyService:
    def __init__(self):
        self.gemini_response_handler = GeminiResponseHandler()

    def _convert_anthropic_to_gemini_format(
        self, anthropic_request: MessagesRequest
    ) -> tuple[str, Dict[str, Any]]:
        """Convert Anthropic MessagesRequest to Gemini format payload.

        Returns:
            tuple: (model_name, payload_dict)
        """
        from app.core.constants import GEMINI_2_FLASH_EXP_SAFETY_SETTINGS

        # Get the actual Gemini model name
        # If model has gemini/ prefix, remove it. Otherwise, it's already the model name
        gemini_model = anthropic_request.model
        if gemini_model.startswith("gemini/"):
            gemini_model = gemini_model.replace("gemini/", "")

        # Convert messages to Gemini contents
        contents = []
        for msg in anthropic_request.messages:
            parts = []
            if isinstance(msg.content, str):
                parts.append({"text": msg.content})
            else:
                for block in msg.content:
                    if isinstance(block, ContentBlockText):
                        parts.append({"text": block.text})
                    elif isinstance(block, ContentBlockImage):
                        # Convert image to Gemini format
                        if "data" in block.source and "media_type" in block.source:
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": block.source["media_type"],
                                        "data": block.source["data"],
                                    }
                                }
                            )
                    elif isinstance(block, ContentBlockToolUse):
                        # Convert tool_use to function call format
                        parts.append(
                            {"function_call": {"name": block.name, "args": block.input}}
                        )
                    elif isinstance(block, ContentBlockToolResult):
                        # Convert tool_result to function response format
                        response_content = block.content
                        if isinstance(response_content, (list, dict)):
                            try:
                                response_content = json.dumps(response_content)
                            except Exception:
                                response_content = str(response_content)
                        parts.append(
                            {
                                "function_response": {
                                    "name": "",  # Gemini doesn't require name
                                    "response": response_content,
                                }
                            }
                        )

            if parts:
                contents.append({"role": msg.role, "parts": parts})

        # Convert system message
        system_instruction = None
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                system_instruction = {
                    "role": "system",
                    "parts": [{"text": anthropic_request.system}],
                }
            elif isinstance(anthropic_request.system, list):
                system_parts = []
                for block in anthropic_request.system:
                    if block.type == "text":
                        system_parts.append({"text": block.text})
                    # Handle other types if needed in the future
                    # For now, only text is supported by Anthropic API
                if system_parts:
                    system_instruction = {"role": "system", "parts": system_parts}

        # Convert tools
        tools = None
        if anthropic_request.tools:
            function_declarations = []
            for tool in anthropic_request.tools:
                input_schema = clean_gemini_schema(
                    tool.input_schema.copy()
                    if isinstance(tool.input_schema, dict)
                    else tool.input_schema
                )
                function_declarations.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": input_schema,
                    }
                )
            tools = [{"functionDeclarations": function_declarations}]

        # Build generation config
        generation_config = {
            "temperature": anthropic_request.temperature,
            "topP": anthropic_request.top_p,
            "topK": anthropic_request.top_k,
            "maxOutputTokens": anthropic_request.max_tokens,
        }
        if anthropic_request.stop_sequences:
            generation_config["stopSequences"] = anthropic_request.stop_sequences

        # Get safety settings
        safety_settings = None
        if gemini_model == "gemini-2.0-flash-exp":
            safety_settings = GEMINI_2_FLASH_EXP_SAFETY_SETTINGS
        elif hasattr(settings, "SAFETY_SETTINGS"):
            safety_settings = settings.SAFETY_SETTINGS

        # Build payload
        payload = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if tools:
            payload["tools"] = tools
        if safety_settings:
            payload["safetySettings"] = safety_settings

        return gemini_model, payload

    def _convert_anthropic_to_litellm(
        self, anthropic_request: MessagesRequest
    ) -> Dict[str, Any]:
        """Convert Anthropic MessagesRequest to LiteLLM format."""
        # Validate request
        if not anthropic_request.messages:
            raise HTTPException(status_code=400, detail="Messages list cannot be empty")

        # Cap max_tokens for OpenAI models to their limit of 16384
        max_tokens = anthropic_request.max_tokens
        is_openai_model = anthropic_request.model.startswith("openai/")
        if is_openai_model:
            max_tokens = min(max_tokens, 16384)
            if anthropic_request.max_tokens > 16384:
                logger.debug(
                    f"Capping max_tokens to 16384 for OpenAI model (original value: {anthropic_request.max_tokens})"
                )

        messages = []
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                messages.append({"role": "system", "content": anthropic_request.system})
            elif isinstance(anthropic_request.system, list):
                system_parts = []
                for block in anthropic_request.system:
                    if block.type == "text":
                        system_parts.append(block.text)
                    # Handle other types if needed in the future
                    # For now, only text is supported by Anthropic API
                if system_parts:
                    system_text = "".join(system_parts)
                    messages.append({"role": "system", "content": system_text})

        for msg in anthropic_request.messages:
            if isinstance(msg.content, str):
                # Ensure content is not empty for OpenAI models
                content = msg.content if msg.content else "..."
                messages.append({"role": msg.role, "content": content})
            else:
                parts = []
                text_content = ""
                has_tool_result = False

                for block in msg.content:
                    if isinstance(block, ContentBlockText):
                        text_content += block.text + "\n"
                    elif isinstance(block, ContentBlockImage):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                                },
                            }
                        )
                    elif isinstance(block, ContentBlockToolUse):
                        parts.append(
                            {
                                "type": "tool_call",
                                "id": block.id,
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                },
                            }
                        )
                    elif isinstance(block, ContentBlockToolResult):
                        has_tool_result = True
                        # For OpenAI models, convert tool_result to text format
                        if is_openai_model:
                            tool_result_text = self._parse_tool_result_content(
                                block.content
                            )
                            text_content += f"Tool result for {block.tool_use_id}:\n{tool_result_text}\n"
                        else:
                            parts.append(
                                {
                                    "type": "tool",
                                    "tool_call_id": block.tool_use_id,
                                    "content": block.content,
                                }
                            )

                # For OpenAI models with tool_result, combine text content
                if is_openai_model and has_tool_result:
                    # Add combined text content
                    if text_content.strip():
                        messages.append(
                            {"role": msg.role, "content": text_content.strip()}
                        )
                    else:
                        messages.append({"role": msg.role, "content": "..."})
                else:
                    # Add text content as a part if we have any
                    if text_content.strip():
                        parts.insert(0, {"type": "text", "text": text_content.strip()})

                    # Ensure we have at least one part
                    if not parts:
                        parts.append({"type": "text", "text": "..."})

                    messages.append({"role": msg.role, "content": parts})

        litellm_request = {
            "model": anthropic_request.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": anthropic_request.temperature,
            "stream": anthropic_request.stream,
        }

        if anthropic_request.tools:
            openai_tools = []
            is_gemini_model = anthropic_request.model.startswith("gemini/")
            for tool in anthropic_request.tools:
                input_schema = (
                    tool.input_schema.copy()
                    if isinstance(tool.input_schema, dict)
                    else tool.input_schema
                )
                if is_gemini_model:
                    input_schema = clean_gemini_schema(input_schema)
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": input_schema,
                        },
                    }
                )
            litellm_request["tools"] = openai_tools

        # Handle tool_choice
        if anthropic_request.tool_choice:
            if isinstance(anthropic_request.tool_choice, dict):
                choice_type = anthropic_request.tool_choice.get("type")
                if choice_type == "auto":
                    litellm_request["tool_choice"] = "auto"
                elif choice_type == "any":
                    litellm_request["tool_choice"] = "any"
                elif choice_type == "tool" and "name" in anthropic_request.tool_choice:
                    litellm_request["tool_choice"] = {
                        "type": "function",
                        "function": {"name": anthropic_request.tool_choice["name"]},
                    }

        # Handle stop_sequences
        if anthropic_request.stop_sequences:
            litellm_request["stop"] = anthropic_request.stop_sequences

        # Handle top_p and top_k
        if anthropic_request.top_p is not None:
            litellm_request["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None:
            litellm_request["top_k"] = anthropic_request.top_k

        return litellm_request

    def _parse_tool_result_content(
        self, content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]
    ) -> str:
        """Parse tool result content into a string format."""
        if content is None:
            return "No content provided"

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            result = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    result += item.get("text", "") + "\n"
                elif isinstance(item, str):
                    result += item + "\n"
                elif isinstance(item, dict):
                    if "text" in item:
                        result += item.get("text", "") + "\n"
                    else:
                        try:
                            result += json.dumps(item) + "\n"
                        except Exception:
                            result += str(item) + "\n"
                else:
                    try:
                        result += str(item) + "\n"
                    except Exception:
                        result += "Unparseable content\n"
            return result.strip()

        if isinstance(content, dict):
            if content.get("type") == "text":
                return content.get("text", "")
            try:
                return json.dumps(content)
            except Exception:
                return str(content)

        # Fallback for any other type
        try:
            return str(content)
        except Exception:
            return "Unparseable content"

    def _from_gemini_to_anthropic(
        self, gemini_response: Dict[str, Any], original_request: MessagesRequest
    ) -> MessagesResponse:
        """Convert Gemini API response to Anthropic format."""
        try:
            # Use the response handler to process Gemini response
            # Get actual model name (remove gemini/ prefix if present)
            actual_model = (
                original_request.model.replace("gemini/", "")
                if original_request.model.startswith("gemini/")
                else original_request.model
            )
            processed_response = self.gemini_response_handler.handle_response(
                gemini_response, actual_model, stream=False
            )

            # Extract data from processed response (OpenAI format)
            choice = processed_response.get("choices", [{}])[0]
            message = choice.get("message", {})
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls")
            finish_reason = choice.get("finish_reason")
            usage_info = processed_response.get("usage", {})
            response_id = processed_response.get("id", f"msg_{uuid.uuid4()}")

            content = []
            if content_text:
                content.append(ContentBlockText(type="text", text=content_text))

            if tool_calls:
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    try:
                        arguments = json.loads(function.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        arguments = {"raw": function.get("arguments", "")}

                    content.append(
                        ContentBlockToolUse(
                            type="tool_use",
                            id=tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                            name=function.get("name", ""),
                            input=arguments,
                        )
                    )

            stop_reason_map: Dict[
                str, Literal["end_turn", "max_tokens", "tool_use"]
            ] = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
            }
            stop_reason = stop_reason_map.get(finish_reason, "end_turn")

            if not content:
                content.append(ContentBlockText(type="text", text=""))

            return MessagesResponse(
                id=response_id,
                model=original_request.original_model or original_request.model,
                content=content,
                stop_reason=stop_reason,
                usage=Usage(
                    input_tokens=usage_info.get("prompt_tokens", 0),
                    output_tokens=usage_info.get("completion_tokens", 0),
                ),
            )
        except Exception as e:
            logger.error(
                f"Error converting Gemini response to anthropic: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500, detail="Error converting response from Gemini API."
            )

    def _convert_litellm_to_anthropic(
        self,
        litellm_response: Union[Dict[str, Any], Any],
        original_request: MessagesRequest,
    ) -> MessagesResponse:
        try:
            if not isinstance(litellm_response, dict):
                choice = litellm_response.choices[0]
                message = choice.message
                content_text = message.content or ""
                tool_calls = message.tool_calls
                finish_reason = choice.finish_reason
                usage_info = litellm_response.usage
                response_id = litellm_response.id
            else:
                choice = litellm_response["choices"][0]
                message = choice["message"]
                content_text = message.get("content", "")
                tool_calls = message.get("tool_calls")
                finish_reason = choice.get("finish_reason")
                usage_info = litellm_response.get("usage", {})
                response_id = litellm_response.get("id", f"msg_{uuid.uuid4()}")

            content = []
            if content_text:
                content.append(ContentBlockText(type="text", text=content_text))

            if tool_calls:
                for tool_call in tool_calls:
                    # Handle both dict and object formats
                    if isinstance(tool_call, dict):
                        function = tool_call.get("function", {})
                        tool_id = tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                        function_name = (
                            function.get("name", "")
                            if isinstance(function, dict)
                            else getattr(function, "name", "")
                        )
                        function_args = (
                            function.get("arguments", "")
                            if isinstance(function, dict)
                            else getattr(function, "arguments", "")
                        )
                    else:
                        function = tool_call.function
                        tool_id = tool_call.id
                        function_name = (
                            function.name if hasattr(function, "name") else ""
                        )
                        function_args = (
                            function.arguments if hasattr(function, "arguments") else ""
                        )

                    try:
                        arguments = (
                            json.loads(function_args)
                            if isinstance(function_args, str)
                            else function_args
                        )
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"raw": str(function_args)} if function_args else {}

                    content.append(
                        ContentBlockToolUse(
                            type="tool_use",
                            id=tool_id,
                            name=function_name,
                            input=arguments,
                        )
                    )

            stop_reason_map: Dict[
                str, Literal["end_turn", "max_tokens", "tool_use"]
            ] = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
            }
            stop_reason = stop_reason_map.get(finish_reason, "end_turn")

            if not content:
                content.append(ContentBlockText(type="text", text=""))

            return MessagesResponse(
                id=response_id,
                model=original_request.original_model or original_request.model,
                content=content,
                stop_reason=stop_reason,
                usage=Usage(
                    input_tokens=getattr(
                        usage_info, "prompt_tokens", usage_info.get("prompt_tokens", 0)
                    ),
                    output_tokens=getattr(
                        usage_info,
                        "completion_tokens",
                        usage_info.get("completion_tokens", 0),
                    ),
                ),
            )
        except Exception as e:
            logger.error(
                f"Error converting litellm response to anthropic: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail="Error converting response from upstream provider.",
            )

    async def _handle_streaming(
        self, response_generator: Any, original_request: MessagesRequest
    ) -> AsyncGenerator[str, None]:
        """Handle streaming responses and convert to Anthropic SSE format."""
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        message_start_data = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": original_request.original_model or original_request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start_data)}\n\n"

        # Track state for streaming
        text_block_started = False
        text_block_index = 0
        tool_blocks: Dict[int, Dict[str, Any]] = {}  # Track tool blocks by index
        tool_block_index = 0
        accumulated_text = ""
        output_tokens = 0
        has_sent_stop_reason = False
        finish_reason = None

        try:
            # Start text content block
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            text_block_started = True

            yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

            async for chunk in response_generator:
                try:
                    # Handle both dict and object formats for chunk
                    if isinstance(chunk, dict):
                        choices = chunk.get("choices", [])
                    else:
                        choices = chunk.choices if hasattr(chunk, "choices") else []

                    if not choices:
                        continue

                    choice = choices[0]
                    # Handle both dict and object formats for delta
                    if isinstance(choice, dict):
                        delta = choice.get("delta", {})
                    else:
                        delta = choice.delta if hasattr(choice, "delta") else {}

                    # Handle text content
                    content_text = (
                        delta.get("content")
                        if isinstance(delta, dict)
                        else (delta.content if hasattr(delta, "content") else None)
                    )
                    if content_text:
                        accumulated_text += content_text
                        delta_data = {
                            "type": "content_block_delta",
                            "index": text_block_index,
                            "delta": {"type": "text_delta", "text": content_text},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(delta_data)}\n\n"
                        output_tokens += len(
                            content_text.split()
                        )  # Rough token estimate

                    # Handle tool calls
                    if isinstance(delta, dict):
                        tool_calls = delta.get("tool_calls")
                    elif hasattr(delta, "tool_calls") and delta.tool_calls:
                        tool_calls = delta.tool_calls
                    else:
                        tool_calls = None

                    if tool_calls:
                        # Close text block if it was started
                        if text_block_started and accumulated_text:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_index})}\n\n"
                            text_block_started = False

                        # Process each tool call
                        for tool_call in tool_calls:
                            # Get tool call index
                            if isinstance(tool_call, dict):
                                tool_index = tool_call.get("index", tool_block_index)
                                function = tool_call.get("function", {})
                                tool_id = tool_call.get(
                                    "id", f"toolu_{uuid.uuid4().hex[:24]}"
                                )
                            else:
                                tool_index = getattr(
                                    tool_call, "index", tool_block_index
                                )
                                function = getattr(tool_call, "function", None)
                                tool_id = getattr(
                                    tool_call, "id", f"toolu_{uuid.uuid4().hex[:24]}"
                                )

                            # Get function name and arguments
                            if isinstance(function, dict):
                                name = function.get("name", "")
                                arguments = function.get("arguments", "")
                            else:
                                name = getattr(function, "name", "") if function else ""
                                arguments = (
                                    getattr(function, "arguments", "")
                                    if function
                                    else ""
                                )

                            # Start new tool block if not already started
                            if tool_index not in tool_blocks:
                                # Use 1-based index for tool blocks (text is 0, tools start at 1)
                                anthropic_tool_index = len(tool_blocks) + 1
                                tool_blocks[tool_index] = {
                                    "id": tool_id,
                                    "name": name,
                                    "arguments": arguments,
                                    "anthropic_index": anthropic_tool_index,
                                }

                                # Start content_block for tool_use
                                content_block = {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": name,
                                    "input": {},
                                }
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': content_block})}\n\n"

                            # Accumulate arguments
                            if arguments:
                                tool_blocks[tool_index]["arguments"] += arguments

                                # Send delta for tool input
                                anthropic_tool_index = tool_blocks[tool_index][
                                    "anthropic_index"
                                ]
                                delta_data = {
                                    "type": "content_block_delta",
                                    "index": anthropic_tool_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": arguments,
                                    },
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(delta_data)}\n\n"

                    # Handle finish_reason
                    finish_reason_value = (
                        choice.get("finish_reason")
                        if isinstance(choice, dict)
                        else (
                            choice.finish_reason
                            if hasattr(choice, "finish_reason")
                            else None
                        )
                    )
                    if finish_reason_value:
                        finish_reason = finish_reason_value
                        has_sent_stop_reason = True

                except Exception as e:
                    logger.error(
                        f"Error processing streaming chunk: {e}", exc_info=True
                    )
                    # Send error event
                    error_data = {
                        "type": "error",
                        "error": {"type": "server_error", "message": str(e)},
                    }
                    yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
                    break

            # Clean up: close any open blocks
            if text_block_started:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_index})}\n\n"

            # Close all tool blocks
            for tool_data in tool_blocks.values():
                anthropic_index = tool_data["anthropic_index"]
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': anthropic_index})}\n\n"

            # Send final message_delta with stop_reason
            if finish_reason or has_sent_stop_reason:
                stop_reason_map: Dict[
                    str, Literal["end_turn", "max_tokens", "tool_use", "stop_sequence"]
                ] = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                }
                stop_reason = (
                    stop_reason_map.get(finish_reason, "end_turn")
                    if finish_reason
                    else "end_turn"
                )

                usage = {"output_tokens": output_tokens}
                delta_stop_data = {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": usage,
                }
                yield f"event: message_delta\ndata: {json.dumps(delta_stop_data)}\n\n"

            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in streaming handler: {e}", exc_info=True)
            # Send error event
            error_data = {
                "type": "error",
                "error": {"type": "server_error", "message": str(e)},
            }
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "data: [DONE]\n\n"

    def _convert_anthropic_to_gemini_for_count_tokens(
        self, anthropic_request: TokenCountRequest
    ) -> Dict[str, Any]:
        """Convert Anthropic TokenCountRequest to Gemini format for token counting."""
        contents = []

        # Add system instruction if present
        system_instruction = None
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                system_instruction = {
                    "role": "system",
                    "parts": [{"text": anthropic_request.system}],
                }
            elif isinstance(anthropic_request.system, list):
                system_parts = []
                for block in anthropic_request.system:
                    if block.type == "text":
                        system_parts.append({"text": block.text})
                    # Handle other types if needed in the future
                    # For now, only text is supported by Anthropic API
                if system_parts:
                    system_instruction = {"role": "system", "parts": system_parts}

        # Convert messages to Gemini contents format
        for msg in anthropic_request.messages:
            parts = []
            if isinstance(msg.content, str):
                parts.append({"text": msg.content})
            else:
                for block in msg.content:
                    if isinstance(block, ContentBlockText):
                        parts.append({"text": block.text})
                    elif isinstance(block, ContentBlockImage):
                        # Convert image to Gemini format
                        if "data" in block.source and "media_type" in block.source:
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": block.source["media_type"],
                                        "data": block.source["data"],
                                    }
                                }
                            )
                    elif isinstance(block, ContentBlockToolUse):
                        # Convert tool_use to function call format
                        parts.append(
                            {"function_call": {"name": block.name, "args": block.input}}
                        )
                    elif isinstance(block, ContentBlockToolResult):
                        # Convert tool_result to function response format
                        parts.append(
                            {
                                "function_response": {
                                    "name": "",  # Gemini doesn't require name in function_response
                                    "response": (
                                        block.content
                                        if isinstance(block.content, (str, dict))
                                        else json.dumps(block.content)
                                    ),
                                }
                            }
                        )

            if parts:
                contents.append({"role": msg.role, "parts": parts})

        payload = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        return payload

    def _convert_anthropic_to_litellm_for_count_tokens(
        self, anthropic_request: TokenCountRequest
    ) -> Dict[str, Any]:
        """Convert Anthropic TokenCountRequest to LiteLLM format for token counting."""
        messages = []

        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                messages.append({"role": "system", "content": anthropic_request.system})
            elif isinstance(anthropic_request.system, list):
                system_parts = []
                for block in anthropic_request.system:
                    if block.type == "text":
                        system_parts.append(block.text)
                    # Handle other types if needed in the future
                    # For now, only text is supported by Anthropic API
                if system_parts:
                    system_text = "".join(system_parts)
                    messages.append({"role": "system", "content": system_text})

        for msg in anthropic_request.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            else:
                parts = []
                for block in msg.content:
                    if isinstance(block, ContentBlockText):
                        parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, ContentBlockImage):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                                },
                            }
                        )
                    elif isinstance(block, ContentBlockToolUse):
                        parts.append(
                            {
                                "type": "tool_call",
                                "id": block.id,
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                },
                            }
                        )
                    elif isinstance(block, ContentBlockToolResult):
                        parts.append(
                            {
                                "type": "tool",
                                "tool_call_id": block.tool_use_id,
                                "content": block.content,
                            }
                        )
                messages.append({"role": msg.role, "content": parts})

        litellm_request = {
            "model": anthropic_request.model,
            "messages": messages,
        }

        if anthropic_request.tools:
            openai_tools = []
            for tool in anthropic_request.tools:
                input_schema = tool.input_schema
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": input_schema,
                        },
                    }
                )
            litellm_request["tools"] = openai_tools

        return litellm_request

    async def count_tokens(
        self,
        request: TokenCountRequest,
        fastapi_request: Optional[Request] = None,
        *,
        session: AsyncSession,
    ) -> TokenCountResponse:
        """Count tokens for the given request using appropriate API client."""
        if not fastapi_request or not hasattr(fastapi_request.app.state, "key_manager"):
            raise RuntimeError("KeyManager not initialized. Request object required.")

        key_manager = fastapi_request.app.state.key_manager
        api_key_info = await key_manager.get_key(
            model_name=request.model, is_vertex_key=False
        )

        if not api_key_info:
            raise HTTPException(
                status_code=429, detail="All API keys are currently exhausted."
            )

        start_time = time.perf_counter()
        request_datetime = datetime.datetime.now()
        is_success = False
        status_code = None
        input_tokens = 0

        try:
            # Route to appropriate API client based on model prefix
            # Check if model is a Gemini model (either has gemini/ prefix or is a known Gemini model name)
            is_gemini_model = (
                request.model.startswith("gemini/")
                or request.model
                in [settings.CLAUDE_SMALL_MODEL, settings.CLAUDE_BIG_MODEL]
                or any(
                    request.model.startswith(gm)
                    for gm in ["gemini-", "gemini-1", "gemini-2"]
                )
            )
            if is_gemini_model:
                # Use GeminiApiClient for Gemini models
                gemini_model = (
                    request.model.replace("gemini/", "")
                    if request.model.startswith("gemini/")
                    else request.model
                )
                gemini_payload = self._convert_anthropic_to_gemini_for_count_tokens(
                    request
                )

                api_client = GeminiApiClient(
                    base_url=settings.BASE_URL, timeout=settings.TIME_OUT
                )
                response = await api_client.count_tokens(
                    gemini_payload, gemini_model, api_key_info
                )

                # Extract token count from Gemini response
                input_tokens = response.get("totalTokens", 0)
                is_success = True
                status_code = 200

            elif request.model.startswith("openai/") or request.model.startswith(
                "anthropic/"
            ):
                # Use LiteLLM token_counter for OpenAI and Anthropic models
                litellm_request = self._convert_anthropic_to_litellm_for_count_tokens(
                    request
                )
                litellm_request["api_key"] = api_key_info

                try:
                    from litellm import token_counter

                    input_tokens = token_counter(
                        model=request.model,
                        messages=litellm_request["messages"],
                    )
                    is_success = True
                    status_code = 200
                except ImportError:
                    logger.error(
                        "Could not import token_counter from litellm", exc_info=True
                    )
                    # Fallback to approximation
                    input_tokens = 1000
                    is_success = True
                    status_code = 200
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported model prefix: {request.model}. Must start with 'gemini/', 'openai/', or 'anthropic/'",
                )

            return TokenCountResponse(input_tokens=input_tokens)

        except HTTPException:
            raise
        except Exception as e:
            is_success = False
            status_code = 500
            error_log_msg = str(e)

            if isinstance(e, ApiClientException):
                status_code = e.status_code
                error_log_msg = e.detail if hasattr(e, "detail") else str(e)
            elif hasattr(e, "args") and len(e.args) > 1:
                status_code = e.args[0] if isinstance(e.args[0], int) else 500
                error_log_msg = e.args[1] if len(e.args) > 1 else str(e)

            logger.error(
                f"Count tokens API call failed: {error_log_msg}", exc_info=True
            )

            await add_error_log(
                session,
                gemini_key=api_key_info,
                model_name=request.model,
                error_type="claude-proxy-count-tokens",
                error_log=error_log_msg,
                error_code=status_code,
                request_msg=(
                    request.model_dump()
                    if settings.ERROR_LOG_RECORD_REQUEST_BODY
                    else None
                ),
                request_datetime=request_datetime,
            )

            await key_manager.handle_api_failure(
                api_key=api_key_info,
                model_name=request.model,
                retries=0,
                status_code=status_code,
            )

            raise HTTPException(
                status_code=status_code,
                detail=f"Error counting tokens: {error_log_msg}",
            )
        finally:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            await add_request_log(
                session,
                model_name=request.model,
                api_key=api_key_info,
                is_success=is_success,
                status_code=status_code,
                latency_ms=latency_ms,
                request_time=request_datetime,
            )

    async def create_message(
        self,
        request: MessagesRequest,
        fastapi_request: Optional[Request] = None,
        *,
        session: AsyncSession,
    ):
        """Create a message using the Claude proxy."""
        if not fastapi_request or not hasattr(fastapi_request.app.state, "key_manager"):
            raise RuntimeError("KeyManager not initialized. Request object required.")

        key_manager = fastapi_request.app.state.key_manager
        api_key_info = await key_manager.get_key(
            model_name=request.model, is_vertex_key=False
        )

        if not api_key_info:
            raise HTTPException(
                status_code=429, detail="All API keys are currently exhausted."
            )

        start_time = time.perf_counter()
        request_datetime = datetime.datetime.now()
        is_success = False
        status_code = None

        try:
            # Route to appropriate API client based on model prefix
            # Check if model is a Gemini model (either has gemini/ prefix or is a known Gemini model name)
            is_gemini_model = (
                request.model.startswith("gemini/")
                or request.model
                in [settings.CLAUDE_SMALL_MODEL, settings.CLAUDE_BIG_MODEL]
                or any(
                    request.model.startswith(gm)
                    for gm in ["gemini-", "gemini-1", "gemini-2"]
                )
            )
            if is_gemini_model:
                # Use GeminiApiClient for Gemini models
                gemini_model, gemini_payload = self._convert_anthropic_to_gemini_format(
                    request
                )
                api_client = GeminiApiClient(
                    base_url=settings.BASE_URL, timeout=settings.TIME_OUT
                )

                if request.stream:
                    # For streaming, convert Gemini SSE to Anthropic SSE
                    async def gemini_to_anthropic_stream():
                        async for line in api_client.stream_generate_content(
                            gemini_payload, gemini_model, api_key_info
                        ):
                            if line.startswith("data:"):
                                line = line[6:].strip()
                                if line and line != "[DONE]":
                                    try:
                                        gemini_chunk = json.loads(line)
                                        # Process through response handler to get OpenAI format
                                        processed_chunk = self.gemini_response_handler.handle_response(
                                            gemini_chunk, gemini_model, stream=True
                                        )

                                        # Convert to LiteLLM-like format for _handle_streaming
                                        # The processed_chunk is a dict with OpenAI format
                                        class MockChunk:
                                            def __init__(self, data):
                                                choices_data = data.get("choices", [{}])
                                                self.choices = [
                                                    (
                                                        MockChoice(choices_data[0])
                                                        if choices_data
                                                        else MockChoice({})
                                                    )
                                                ]

                                        class MockChoice:
                                            def __init__(self, data):
                                                delta_data = data.get("delta", {})
                                                self.delta = MockDelta(delta_data)
                                                self.finish_reason = data.get(
                                                    "finish_reason"
                                                )

                                        class MockDelta:
                                            def __init__(self, data):
                                                self.content = data.get("content")
                                                self.tool_calls = data.get("tool_calls")

                                        yield MockChunk(processed_chunk)
                                    except (
                                        json.JSONDecodeError,
                                        KeyError,
                                        IndexError,
                                    ) as e:
                                        logger.debug(
                                            f"Error processing streaming chunk: {e}"
                                        )
                                        continue

                    # Wrap the generator for _handle_streaming
                    response_generator = gemini_to_anthropic_stream()
                    is_success = True
                    status_code = 200
                    return StreamingResponse(
                        self._handle_streaming(response_generator, request),  # type: ignore
                        media_type="text/event-stream",
                    )
                else:
                    # Non-streaming
                    gemini_response = await api_client.generate_content(
                        gemini_payload, gemini_model, api_key_info
                    )
                    response = self._from_gemini_to_anthropic(gemini_response, request)

                    # Update usage stats
                    if "usageMetadata" in gemini_response:
                        await update_usage_stats(
                            session,
                            api_key=api_key_info,
                            model_name=request.model,
                            token_count=gemini_response["usageMetadata"].get(
                                "totalTokenCount", 0
                            ),
                            tpm=gemini_response["usageMetadata"].get(
                                "totalTokenCount", 0
                            ),
                        )

                    is_success = True
                    status_code = 200
                    return response
            else:
                # Use LiteLLM for OpenAI and Anthropic models
                litellm_request = self._convert_anthropic_to_litellm(request)
                litellm_request["api_key"] = api_key_info

                if request.stream:
                    response_generator = await litellm.acompletion(**litellm_request)
                    # For streaming, we can't log in finally block easily, so we log here
                    is_success = True
                    status_code = 200
                    return StreamingResponse(
                        self._handle_streaming(response_generator, request),  # type: ignore
                        media_type="text/event-stream",
                    )
                else:
                    litellm_response = await litellm.acompletion(**litellm_request)
                    response = self._convert_litellm_to_anthropic(
                        litellm_response, request
                    )
                    is_success = True
                    status_code = 200
                    return response

        except HTTPException:
            raise
        except Exception as e:
            is_success = False
            status_code = 500
            error_log_msg = str(e)

            if isinstance(e, ApiClientException):
                status_code = e.status_code
                error_log_msg = e.detail if hasattr(e, "detail") else str(e)
            elif hasattr(e, "args") and len(e.args) > 1:
                status_code = e.args[0] if isinstance(e.args[0], int) else 500
                error_log_msg = e.args[1] if len(e.args) > 1 else str(e)
            elif hasattr(e, "args") and len(e.args) > 0 and isinstance(e.args[0], int):
                status_code = e.args[0]

            logger.error(f"Error calling litellm: {error_log_msg}", exc_info=True)

            await add_error_log(
                session,
                gemini_key=api_key_info,
                model_name=request.model,
                error_type="claude-proxy-messages",
                error_log=error_log_msg,
                error_code=status_code,
                request_msg=(
                    request.model_dump()
                    if settings.ERROR_LOG_RECORD_REQUEST_BODY
                    else None
                ),
                request_datetime=request_datetime,
            )

            await key_manager.handle_api_failure(
                api_key=api_key_info,
                model_name=request.model,
                retries=0,
                status_code=status_code,
            )

            raise HTTPException(
                status_code=status_code,
                detail=f"Error creating message: {error_log_msg}",
            )
        finally:
            # Only log non-streaming requests in finally block
            if not request.stream:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                await add_request_log(
                    session,
                    model_name=request.model,
                    api_key=api_key_info,
                    is_success=is_success,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    request_time=request_datetime,
                )
