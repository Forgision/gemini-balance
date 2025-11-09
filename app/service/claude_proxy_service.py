"""
Claude Proxy Service
"""
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

import litellm
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from app.config.config import settings
from app.log.logger import get_gemini_logger
from app.service.key.key_manager import get_key_manager_instance

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
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
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
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool = True

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

    @field_validator('model')
    def validate_model_field(cls, v, info):
        original_model = v
        new_model = v

        clean_v = v.split('/')[-1]

        mapped = False
        if settings.CLAUDE_PREFERRED_PROVIDER == "anthropic":
            new_model = f"anthropic/{clean_v}"
            mapped = True
        elif 'haiku' in clean_v.lower():
            new_model = f"gemini/{settings.CLAUDE_SMALL_MODEL}"
            mapped = True
        elif 'sonnet' in clean_v.lower():
            new_model = f"gemini/{settings.CLAUDE_BIG_MODEL}"
            mapped = True

        if mapped:
            logger.debug(f"Model mapping: '{original_model}' -> '{new_model}'")
        else:
            if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                logger.warning(f"No prefix or mapping rule for model: '{original_model}'. Using as is.")
            new_model = v

        if 'values' in info and isinstance(info.data, dict):
            info.data['original_model'] = original_model

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
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

class ClaudeProxyService:
    def __init__(self):
        pass

    def _convert_anthropic_to_litellm(self, anthropic_request: MessagesRequest) -> Dict[str, Any]:
        messages = []
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                messages.append({"role": "system", "content": anthropic_request.system})
            elif isinstance(anthropic_request.system, list):
                system_text = "".join(block.text for block in anthropic_request.system if block.type == "text")
                if system_text:
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
                        parts.append({"type": "image_url", "image_url": {"url": f"data:{block.source['media_type']};base64,{block.source['data']}"}})
                    elif isinstance(block, ContentBlockToolUse):
                        parts.append({
                            "type": "tool_call",
                            "id": block.id,
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input)
                            }
                        })
                    elif isinstance(block, ContentBlockToolResult):
                        parts.append({
                            "type": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": block.content
                        })
                messages.append({"role": msg.role, "content": parts})

        litellm_request = {
            "model": anthropic_request.model,
            "messages": messages,
            "max_tokens": anthropic_request.max_tokens,
            "temperature": anthropic_request.temperature,
            "stream": anthropic_request.stream,
        }

        if anthropic_request.tools:
            openai_tools = []
            is_gemini_model = anthropic_request.model.startswith("gemini/")
            for tool in anthropic_request.tools:
                input_schema = tool.input_schema
                if is_gemini_model:
                    input_schema = clean_gemini_schema(input_schema)
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": input_schema
                    }
                })
            litellm_request["tools"] = openai_tools

        return litellm_request

    def _convert_litellm_to_anthropic(self, litellm_response: Union[Dict[str, Any], Any], original_request: MessagesRequest) -> MessagesResponse:
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
                    function = tool_call.function
                    try:
                        arguments = json.loads(function.arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": function.arguments}

                    content.append(ContentBlockToolUse(
                        type="tool_use",
                        id=tool_call.id,
                        name=function.name,
                        input=arguments
                    ))

            stop_reason_map: Dict[str, Literal["end_turn", "max_tokens", "tool_use"]] = {
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
                    input_tokens=getattr(usage_info, 'prompt_tokens', usage_info.get('prompt_tokens', 0)),
                    output_tokens=getattr(usage_info, 'completion_tokens', usage_info.get('completion_tokens', 0))
                )
            )
        except Exception as e:
            logger.error(f"Error converting litellm response to anthropic: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error converting response from upstream provider.")

    async def _handle_streaming(self, response_generator: Any, original_request: MessagesRequest) -> AsyncGenerator[str, None]:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        message_start_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.original_model or original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 0, 'output_tokens': 0}
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_start_data)}\n\n"

        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        output_tokens = 0
        async for chunk in response_generator:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                delta_data = {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta.content}}
                yield f"event: content_block_delta\ndata: {json.dumps(delta_data)}\n\n"

            if choice.finish_reason:
                stop_reason_map: Dict[str, Literal["end_turn", "max_tokens", "tool_use"]] = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                }
                stop_reason = stop_reason_map.get(choice.finish_reason, "end_turn")

                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                usage = {"output_tokens": output_tokens}
                delta_stop_data = {'type': 'message_delta', 'delta': {'stop_reason': stop_reason}, 'usage': usage}
                yield f"event: message_delta\ndata: {json.dumps(delta_stop_data)}\n\n"

        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    async def create_message(self, request: MessagesRequest):
        key_manager = await get_key_manager_instance()
        api_key_info = await key_manager.get_next_working_key(model_name=request.model)

        if not api_key_info:
            raise HTTPException(status_code=429, detail="All API keys are currently exhausted.")

        litellm_request = self._convert_anthropic_to_litellm(request)
        litellm_request["api_key"] = api_key_info

        try:
            if request.stream:
                response_generator = await litellm.acompletion(**litellm_request)
                return StreamingResponse(
                    self._handle_streaming(response_generator, request), # type: ignore
                    media_type="text/event-stream"
                )
            else:
                litellm_response = await litellm.acompletion(**litellm_request)
                return self._convert_litellm_to_anthropic(litellm_response, request)
        except Exception as e:
            logger.error(f"Error calling litellm: {e}", exc_info=True)
            await key_manager.handle_api_failure(api_key=api_key_info, model_name=request.model, retries=0) # type: ignore
            raise HTTPException(status_code=500, detail=str(e))
