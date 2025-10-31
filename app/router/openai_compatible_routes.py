from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.config.config import settings
from app.core.security import SecurityService
from app.dependencies import get_key_manager
from app.dependencies import get_openai_compatible_chat_service as get_openai_service
from app.domain.openai_models import (
    ChatRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
)
from app.handler.error_handler import handle_route_errors
from app.handler.retry_handler import RetryHandler
from app.log.logger import get_openai_compatible_logger
from app.service.key.key_manager import KeyManager
from app.service.openai_compatiable.openai_compatiable_service import (
    OpenAICompatiableService,
)
from app.utils.helpers import redact_key_for_logging

router = APIRouter()
logger = get_openai_compatible_logger()

security_service = SecurityService()


async def get_next_working_key_wrapper(
    key_manager: KeyManager = Depends(get_key_manager),
):
    return await key_manager.get_next_working_key(model_name="gemini-pro")


@router.get("/openai/v1/models")
async def list_models(
    allowed_token=Depends(security_service.verify_authorization),
    key_manager: KeyManager = Depends(get_key_manager),
    openai_service: OpenAICompatiableService = Depends(get_openai_service),
):
    """Get the list of available models."""
    operation_name = "list_models"
    async with handle_route_errors(logger, operation_name):
        logger.info("Handling models list request")
        api_key = await key_manager.get_random_valid_key()
        logger.info(f"Using allowed token: {allowed_token}")
        logger.info(f"Using API key: {redact_key_for_logging(api_key)}")
        return await openai_service.get_models(api_key)


@router.post("/openai/v1/chat/completions")
@RetryHandler(key_arg="api_key")
async def chat_completion(
    request: ChatRequest,
    allowed_token=Depends(security_service.verify_authorization),
    api_key: str = Depends(get_next_working_key_wrapper),
    key_manager: KeyManager = Depends(get_key_manager),
    openai_service: OpenAICompatiableService = Depends(get_openai_service),
):
    """Handle chat completion requests, supporting streaming responses and specific model switching."""
    operation_name = "chat_completion"
    is_image_chat = request.model == f"{settings.CREATE_IMAGE_MODEL}-chat"
    current_api_key = api_key
    if is_image_chat:
        current_api_key = await key_manager.get_paid_key()

    async with handle_route_errors(logger, operation_name):
        logger.info(f"Handling chat completion request for model: {request.model}")
        logger.debug(f"Request: \n{request.model_dump_json(indent=2)}")
        logger.info(f"Using allowed token: {allowed_token}")
        logger.info(f"Using API key: {redact_key_for_logging(current_api_key)}")

        raw_response = None
        if is_image_chat:
            raw_response = await openai_service.create_image_chat_completion(
                request, current_api_key
            )
        else:
            raw_response = await openai_service.create_chat_completion(
                request, current_api_key
            )
        if request.stream:
            # Check if raw_response is a dictionary (which indicates an error), if so, return it directly.
            if isinstance(raw_response, dict):
                return JSONResponse(content=raw_response, status_code=500)
            try:
                # Try to get the first piece of data to determine if it is a normal SSE (data: prefix) or an error JSON
                first_chunk = await raw_response.__anext__()
            except StopAsyncIteration:
                # If the stream ends directly, return a standard SSE output
                return StreamingResponse(
                    (c for c in []), media_type="text/event-stream"
                )
            except Exception as e:
                # Initialization stream exception, return a 500 error directly
                return JSONResponse(
                    content={"error": {"code": e.args[0], "message": e.args[1]}},
                    status_code=e.args[0],
                )

            # If it starts with "data:", it means it is a normal SSE, and the first block and subsequent blocks will be sent together
            if isinstance(first_chunk, str) and first_chunk.startswith("data:"):

                async def combined():
                    yield first_chunk
                    async for chunk in raw_response:
                        yield chunk

                return StreamingResponse(combined(), media_type="text/event-stream")
        else:
            return raw_response


@router.post("/openai/v1/images/generations")
async def generate_image(
    request: ImageGenerationRequest,
    allowed_token=Depends(security_service.verify_authorization),
    key_manager: KeyManager = Depends(get_key_manager),
    openai_service: OpenAICompatiableService = Depends(get_openai_service),
):
    """Handle image generation requests."""
    operation_name = "generate_image"
    async with handle_route_errors(logger, operation_name):
        logger.info(f"Handling image generation request for prompt: {request.prompt}")
        logger.info(f"Using allowed token: {allowed_token}")
        request.model = settings.CREATE_IMAGE_MODEL
        api_key = await key_manager.get_paid_key()
        return await openai_service.generate_images(request, api_key)


@router.post("/openai/v1/embeddings")
async def embedding(
    request: EmbeddingRequest,
    allowed_token=Depends(security_service.verify_authorization),
    key_manager: KeyManager = Depends(get_key_manager),
    openai_service: OpenAICompatiableService = Depends(get_openai_service),
):
    """Handle text embedding requests."""
    operation_name = "embedding"
    async with handle_route_errors(logger, operation_name):
        logger.info(f"Handling embedding request for model: {request.model}")
        api_key = await key_manager.get_next_working_key(model_name=request.model)
        logger.info(f"Using allowed token: {allowed_token}")
        logger.info(f"Using API key: {redact_key_for_logging(api_key)}")
        input_text = (
            request.input if isinstance(request.input, str) else " ".join(request.input)
        )
        return await openai_service.create_embeddings(
            input_text=input_text, model=request.model, api_key=api_key
        )
