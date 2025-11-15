import inspect
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from app.config.config import settings
from app.exception.api_exceptions import ApiClientException
from app.log.logger import get_retry_logger
from app.utils.helpers import redact_key_for_logging

T = TypeVar("T")
logger = get_retry_logger()


def RetryHandler(key_arg: str = "api_key", model_arg: str = "model_name"):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            key_manager = kwargs.get("key_manager")
            if not key_manager:
                # Try to get from request if available
                # Check for both 'request' and 'fastapi_request' (some routes use different names)
                fastapi_request = kwargs.get("fastapi_request") or kwargs.get("request")
                # Only try to access .app if it's a FastAPI Request object (has 'app' attribute)
                if fastapi_request and hasattr(fastapi_request, "app") and hasattr(fastapi_request.app.state, "key_manager"):
                    key_manager = fastapi_request.app.state.key_manager
                else:
                    raise ValueError("KeyManager instance is not available. Use dependency injection or provide request object.")

            if key_manager is None:
                raise ValueError("KeyManager instance is not available.")

            last_exception = None

            for attempt in range(settings.MAX_RETRIES):
                retries = attempt + 1
                try:
                    if inspect.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"API call failed with error: {str(e)}. Attempt {retries} of {settings.MAX_RETRIES}"
                    )

                    old_key = kwargs.get(key_arg)
                    if not old_key:
                        logger.error("No API key found in arguments for retry handler.")
                        break

                    # TODO: test that model_name works with routes from openai_routes.py and vertext_express_routes.py
                    model_name = kwargs.get(model_arg)
                    if not model_name:
                        logger.error(
                            "No model name found in arguments for retry handler."
                        )
                        break

                    # Extract status_code from exception
                    status_code = None
                    if isinstance(e, ApiClientException):
                        status_code = e.status_code
                    elif hasattr(e, 'args') and len(e.args) > 0:
                        status_code = e.args[0] if isinstance(e.args[0], int) else None
                    if status_code is None:
                        status_code = 500  # Default to 500 if not available

                    new_key = await key_manager.handle_api_failure(
                        old_key, model_name, retries, status_code=status_code
                    )
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
                last_exception = RuntimeError(
                    "Retry handler failed without an exception."
                )
            logger.error(
                f"All retry attempts failed, raising final exception: {str(last_exception)}"
            )
            raise last_exception

        return cast(Callable[..., T], wrapper)

    return decorator
