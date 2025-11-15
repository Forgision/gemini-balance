"""
Claude Proxy Routes
"""

from fastapi import APIRouter, Depends, Request, HTTPException
from starlette.requests import Request
from app.service.claude_proxy_service import ClaudeProxyService, MessagesRequest, TokenCountRequest
from app.core.security import SecurityService
from app.dependencies import get_key_manager
from app.handler.error_handler import handle_route_errors
from app.handler.retry_handler import RetryHandler
from app.log.logger import get_gemini_logger
from app.service.key.key_manager import KeyManager
from app.utils.helpers import redact_key_for_logging

logger = get_gemini_logger()

# Security service instance
security_service = SecurityService()

# Router for /claude/v1/ prefix (works with Claude Code CLI)
router = APIRouter(prefix="/claude/v1", tags=["Claude Proxy"])


async def get_next_working_key(
    request: MessagesRequest,
    key_manager: KeyManager = Depends(get_key_manager),
) -> str:
    """Get the next available API key for Claude proxy requests."""
    return await key_manager.get_key(model_name=request.model, is_vertex_key=False)


async def get_next_working_key_for_token_count(
    request: TokenCountRequest,
    key_manager: KeyManager = Depends(get_key_manager),
) -> str:
    """Get the next available API key for token count requests."""
    return await key_manager.get_key(model_name=request.model, is_vertex_key=False)


@router.post("/messages")
@RetryHandler(key_arg="api_key", model_arg="request.model")
async def create_message(
    request: MessagesRequest,
    fastapi_request: Request,
    allowed_token=Depends(security_service.verify_auth_token),
    api_key: str = Depends(get_next_working_key),
    key_manager: KeyManager = Depends(get_key_manager),
    service: ClaudeProxyService = Depends(ClaudeProxyService),
):
    """
    Handle chat completions using the Claude proxy.
    """
    operation_name = "claude_proxy_messages"
    async with handle_route_errors(
        logger, operation_name, failure_message="Claude proxy message creation failed"
    ):
        logger.info(f"Handling Claude proxy message request for model: {request.model}")
        logger.debug(f"Request: \n{request.model_dump_json(indent=2)}")
        logger.info(f"Using allowed token: {allowed_token}")
        logger.info(f"Using API key: {redact_key_for_logging(api_key)}")
        return await service.create_message(request, fastapi_request)


@router.post("/messages/count_tokens")
@RetryHandler(key_arg="api_key", model_arg="request.model")
async def count_tokens(
    request: TokenCountRequest,
    fastapi_request: Request,
    allowed_token=Depends(security_service.verify_auth_token),
    api_key: str = Depends(get_next_working_key_for_token_count),
    key_manager: KeyManager = Depends(get_key_manager),
    service: ClaudeProxyService = Depends(ClaudeProxyService),
):
    """
    Count tokens for the given messages using the Claude proxy.
    """
    operation_name = "claude_proxy_count_tokens"
    async with handle_route_errors(
        logger, operation_name, failure_message="Claude proxy token counting failed"
    ):
        logger.info(f"Handling Claude proxy token count request for model: {request.model}")
        logger.debug(f"Request: \n{request.model_dump_json(indent=2)}")
        logger.info(f"Using allowed token: {allowed_token}")
        logger.info(f"Using API key: {redact_key_for_logging(api_key)}")
        return await service.count_tokens(request, fastapi_request)
