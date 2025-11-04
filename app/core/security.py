from typing import Optional

from fastapi import Header, HTTPException

from app.config.config import settings
from app.log.logger import get_security_logger

logger = get_security_logger()


def verify_auth_token(token: str) -> bool:
    return token == settings.AUTH_TOKEN


class SecurityService:
    async def verify_key(self, key: str):
        if key not in settings.ALLOWED_TOKENS and key != settings.AUTH_TOKEN:
            logger.error("Invalid key")
            raise HTTPException(status_code=401, detail="Invalid key")
        return key

    async def verify_authorization(
        self, authorization: Optional[str] = Header(None)
    ) -> str:
        if not authorization:
            logger.error("Missing Authorization header")
            raise HTTPException(status_code=401, detail="Missing Authorization header")

        if not authorization.startswith("Bearer "):
            logger.error("Invalid Authorization header format")
            raise HTTPException(
                status_code=401, detail="Invalid Authorization header format"
            )

        token = authorization.replace("Bearer ", "")
        if token not in settings.ALLOWED_TOKENS and token != settings.AUTH_TOKEN:
            logger.error("Invalid token")
            raise HTTPException(status_code=401, detail="Invalid token")

        return token

    async def verify_goog_api_key(
        self, x_goog_api_key: Optional[str] = Header(None)
    ) -> str:
        """Verify Google API Key"""
        if not x_goog_api_key:
            logger.error("Missing x-goog-api-key header")
            raise HTTPException(status_code=401, detail="Missing x-goog-api-key header")

        if (
            x_goog_api_key not in settings.ALLOWED_TOKENS
            and x_goog_api_key != settings.AUTH_TOKEN
        ):
            logger.error("Invalid x-goog-api-key")
            raise HTTPException(status_code=401, detail="Invalid x-goog-api-key")

        return x_goog_api_key

    async def verify_auth_token(
        self, authorization: Optional[str] = Header(None)
    ) -> str:
        """
        For OpenAI routes Verifies the provided authorization token from the request header.
        Args:
            authorization (Optional[str]): The 'Authorization' header value from the request, expected in the format 'Bearer <token>'.
        Returns:
            str: The validated token string.
        Raises:
            HTTPException: If the authorization header is missing or the token is invalid.
        """
        
        if not authorization:
            logger.error("Missing auth_token header")
            raise HTTPException(status_code=401, detail="Missing auth_token header")
        token = authorization.replace("Bearer ", "")
        if token != settings.AUTH_TOKEN:
            logger.error("Invalid auth_token")
            raise HTTPException(status_code=401, detail="Invalid auth_token")

        return token

    async def verify_key_or_goog_api_key(
        self, key: Optional[str] = None, x_goog_api_key: Optional[str] = Header(None)
    ) -> str:
        """Verify the key in the URL or the x-goog-api-key in the request header

        This asynchronous dependency inspects an optional `key` (typically supplied via
        a URL query or path parameter) and an optional `x_goog_api_key` header (injected
        via FastAPI's Header). It validates tokens against the application's token
        sources (settings.ALLOWED_TOKENS and settings.AUTH_TOKEN).

        Behavior:
        - If `key` is provided and matches settings.ALLOWED_TOKENS or
            settings.AUTH_TOKEN, it is returned immediately (URL key takes precedence).
        - Otherwise, the function requires a valid `x_goog_api_key` header and returns
            that header value if valid.
        - If neither a valid URL key nor a valid header is present, an HTTP 401 error
            is raised and an error is logged.

        Args:
                key (Optional[str]): Token supplied in the URL (query/path).
                x_goog_api_key (Optional[str]): Token supplied in the 'x-goog-api-key'
                        request header (injected via FastAPI Header).

        Returns:
                str: The validated token (either the URL key or the header token).

        Raises:
                fastapi.HTTPException: With status_code=401 when:
                        - The URL key is missing/invalid and the header is missing
                            ("Invalid key and missing x-goog-api-key header"), or
                        - The URL key is missing/invalid and the header is present but invalid
                            ("Invalid key and invalid x-goog-api-key").

        Notes:
                - Validation is performed against settings.ALLOWED_TOKENS and
                    settings.AUTH_TOKEN.
                - Error details are logged before raising the HTTPException.
        """
        # If the key in the URL is valid, return it directly
        if key is not None and (
            key in settings.ALLOWED_TOKENS or key == settings.AUTH_TOKEN
        ):
            return key

        # Otherwise, check the x-goog-api-key in the request header
        if not x_goog_api_key:
            logger.error("Invalid key and missing x-goog-api-key header")
            raise HTTPException(
                status_code=401, detail="Invalid key and missing x-goog-api-key header"
            )

        if (
            x_goog_api_key not in settings.ALLOWED_TOKENS
            and x_goog_api_key != settings.AUTH_TOKEN
        ):
            logger.error("Invalid key and invalid x-goog-api-key")
            raise HTTPException(
                status_code=401, detail="Invalid key and invalid x-goog-api-key"
            )

        return x_goog_api_key
