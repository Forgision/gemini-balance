import inspect
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from app.config.config import settings
from app.exception.api_exceptions import ApiClientException
from app.log.logger import get_retry_logger
from app.utils.helpers import redact_key_for_logging

T = TypeVar("T")
logger = get_retry_logger()


def RetryHandler(key_arg: str = "api_key", model_arg: str = "model_name", model_arg_required: bool = True):
    """
    Retry handler decorator for API routes with automatic key switching.
    
    Args:
        key_arg: Name of the API key argument in the function signature
        model_arg: Name of the model argument in the function signature. Can be a nested path like "request.model"
        model_arg_required: If False, retry logic will work even without model_name (e.g., for list_models endpoints)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            key_manager = kwargs.get("key_manager")
            if not key_manager:
                # Try to get from request if available
                # Check for both 'request' and 'fastapi_request' (some routes use different names)
                # Only check if these are actually in the function signature to avoid FastAPI validation issues
                func_sig = inspect.signature(func)
                param_names = set(func_sig.parameters.keys())
                
                fastapi_request = None
                if "fastapi_request" in param_names and "fastapi_request" in kwargs:
                    fastapi_request = kwargs.get("fastapi_request")
                elif "request" in param_names and "request" in kwargs:
                    fastapi_request = kwargs.get("request")
                
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

                    # Extract model_name - handle nested paths like "request.model"
                    model_name = None
                    if "." in model_arg:
                        # Handle nested paths like "request.model"
                        parts = model_arg.split(".")
                        obj = kwargs.get(parts[0])
                        for part in parts[1:]:
                            if obj is not None and hasattr(obj, part):
                                obj = getattr(obj, part)
                                if isinstance(obj, str):
                                    model_name = obj
                                    break
                            elif obj is not None and isinstance(obj, dict) and part in obj:
                                obj = obj[part]
                                if isinstance(obj, str):
                                    model_name = obj
                                    break
                            else:
                                break
                    else:
                        model_name = kwargs.get(model_arg)
                    
                    # If model_arg is required but not found, break retry loop
                    if model_arg_required and not model_name:
                        logger.warning(
                            f"No model name found in arguments for retry handler (model_arg='{model_arg}'). "
                            f"Skipping retry logic for this endpoint."
                        )
                        break
                    
                    # If model_name is not available and not required, use a default for key switching
                    if not model_name:
                        # For endpoints without model_name (like list_models), use a default model
                        # This allows key switching but may not be model-specific
                        model_name = "default"
                        logger.debug(
                            "Model name not found, using default for key switching."
                        )

                    # Extract status_code from exception
                    status_code = None
                    if isinstance(e, ApiClientException):
                        status_code = e.status_code
                    elif hasattr(e, 'args') and len(e.args) > 0:
                        status_code = e.args[0] if isinstance(e.args[0], int) else None
                    if status_code is None:
                        status_code = 500  # Default to 500 if not available

                    # Only handle API failure if we have a model_name (even if default)
                    if model_name:
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
                    else:
                        # No model_name available, can't do key switching
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
