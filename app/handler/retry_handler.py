import asyncio
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from fastapi import Depends
from fastapi import Depends
from app.config.config import settings
from app.log.logger import get_retry_logger
from app.service.key.key_manager import KeyManager, get_key_manager_instance
from app.service.key.key_manager import KeyManager, get_key_manager_instance
from app.utils.helpers import redact_key_for_logging

T = TypeVar("T")
logger = get_retry_logger()


def RetryHandler(key_arg: str = "api_key"):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
def RetryHandler(key_arg: str = "api_key"):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            key_manager = kwargs.get("key_manager")
            if not key_manager:
                # Fallback to singleton if not injected
                key_manager = await get_key_manager_instance()

            if key_manager is None:
                raise ValueError("KeyManager instance is not available.")

            last_exception = None

            for attempt in range(settings.MAX_RETRIES):
                retries = attempt + 1
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"API call failed with error: {str(e)}. Attempt {retries} of {settings.MAX_RETRIES}"
                    )

                    old_key = kwargs.get(key_arg)
                    if not old_key:
                        logger.error("No API key found in arguments for retry handler.")
                        break

                    new_key = await key_manager.handle_api_failure(old_key, retries)
                    if new_key:
                        kwargs[key_arg] = new_key
                        logger.info(
                            f"Switched to new API key: {redact_key_for_logging(new_key)}"
                        )
                    else:
                        logger.error(
                            f"No valid API key available after {retries} retries."
                        )
                        break

            if last_exception is None:
                # This should not happen if MAX_RETRIES > 0
                last_exception = RuntimeError("Retry handler failed without an exception.")
            logger.error(
                f"All retry attempts failed, raising final exception: {str(last_exception)}"
            )
            raise last_exception

        return cast(Callable[..., T], wrapper)

    return decorator
