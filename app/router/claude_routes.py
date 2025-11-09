"""
Claude Proxy Routes
"""

from fastapi import APIRouter, Depends
from app.service.claude_proxy_service import ClaudeProxyService, MessagesRequest
from app.core.security import verify_auth_token

router = APIRouter(prefix="/claude/v1", tags=["Claude Proxy"], dependencies=[Depends(verify_auth_token)])

@router.post("/messages")
async def create_message(request: MessagesRequest, service: ClaudeProxyService = Depends(ClaudeProxyService)):
    """
    Handle chat completions using the Claude proxy.
    """
    return await service.create_message(request)
