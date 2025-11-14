"""
Route configuration module, responsible for setting up and configuring the application's routes.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config.config import settings
from app.core.security import verify_auth_token
from app.log.logger import get_routes_logger
from app.router import (
    claude_routes,
    config_routes,
    error_log_routes,
    files_routes,
    gemini_routes,
    key_routes,
    openai_compatible_routes,
    openai_routes,
    scheduler_routes,
    stats_routes,
    usage_stats_routes,
    version_routes,
    vertex_express_routes,
)
# No longer using get_key_manager_instance - use app.state
from app.service.stats.stats_service import StatsService
from app.utils.static_version import get_static_url

logger = get_routes_logger()

templates = Jinja2Templates(directory="app/templates")
# Set template global variables
templates.env.globals["static_url"] = get_static_url


def setup_routers(app: FastAPI) -> None:
    """
    Set up the application's routes.

    Args:
        app: FastAPI application instance
    """
    app.include_router(openai_routes.router)
    app.include_router(gemini_routes.router)
    app.include_router(gemini_routes.router_v1beta)
    app.include_router(config_routes.router)
    app.include_router(error_log_routes.router)
    app.include_router(scheduler_routes.router)
    app.include_router(stats_routes.router)
    app.include_router(version_routes.router)
    app.include_router(openai_compatible_routes.router)
    app.include_router(vertex_express_routes.router)
    app.include_router(files_routes.router)
    app.include_router(key_routes.router)
    app.include_router(usage_stats_routes.router)
    app.include_router(claude_routes.router)

    setup_page_routes(app)

    setup_health_routes(app)
    setup_api_stats_routes(app)


def setup_page_routes(app: FastAPI) -> None:
    """
    Set up page-related routes.

    Args:
        app: FastAPI application instance
    """

    @app.get("/", response_class=HTMLResponse)
    async def auth_page(request: Request):
        """Authentication page"""
        return templates.TemplateResponse("auth.html", {"request": request})

    @app.post("/auth")
    async def authenticate(request: Request):
        """Handle authentication requests."""
        try:
            form = await request.form()
            auth_token = form.get("auth_token")
            if not auth_token or not isinstance(auth_token, str):
                logger.warning("Authentication attempt with empty or invalid token")
                return RedirectResponse(url="/", status_code=302)

            if verify_auth_token(auth_token):
                logger.info("Successful authentication")
                response = RedirectResponse(url="/keys", status_code=302)
                response.set_cookie(
                    key="auth_token",
                    value=auth_token,
                    httponly=True,
                    max_age=settings.ADMIN_SESSION_EXPIRE,
                )
                return response
            logger.warning("Failed authentication attempt with invalid token")
            return RedirectResponse(url="/", status_code=302)
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return RedirectResponse(url="/", status_code=302)

    @app.get("/keys", response_class=HTMLResponse)
    async def keys_page(request: Request):
        """Key management page."""
        try:
            auth_token = request.cookies.get("auth_token")
            if not auth_token or not verify_auth_token(auth_token):
                logger.warning("Unauthorized access attempt to keys page")
                return RedirectResponse(url="/", status_code=302)

            if not hasattr(request.app.state, "key_manager"):
                raise RuntimeError("KeyManager not initialized.")
            key_manager = request.app.state.key_manager
            keys_status = await key_manager.get_keys_by_status()
            total_keys = len(keys_status["valid_keys"]) + len(
                keys_status["invalid_keys"]
            )
            valid_key_count = len(keys_status["valid_keys"])
            invalid_key_count = len(keys_status["invalid_keys"])

            stats_service = StatsService()
            api_stats = await stats_service.get_api_usage_stats()
            logger.info(f"API stats retrieved: {api_stats}")

            logger.info(f"Keys status retrieved successfully. Total keys: {total_keys}")
            return templates.TemplateResponse(
                "keys_status.html",
                {
                    "request": request,
                    "valid_keys": {},
                    "invalid_keys": {},
                    "total_keys": total_keys,
                    "valid_key_count": valid_key_count,
                    "invalid_key_count": invalid_key_count,
                    "api_stats": api_stats,
                },
            )
        except Exception as e:
            logger.error(f"Error retrieving keys status or API stats: {str(e)}")
            # Even if there's an error, render the page with whatever data is available
            # or with empty/default values, so the frontend can still load.
            return templates.TemplateResponse(
                "keys_status.html",
                {
                    "request": request,
                    "valid_keys": {},
                    "invalid_keys": {},
                    "total_keys": 0,
                    "valid_key_count": 0,
                    "invalid_key_count": 0,
                    "api_stats": {  # Provide a default structure for api_stats
                        "calls_1m": {"total": 0, "success": 0, "failure": 0},
                        "calls_1h": {"total": 0, "success": 0, "failure": 0},
                        "calls_24h": {"total": 0, "success": 0, "failure": 0},
                        "calls_month": {"total": 0, "success": 0, "failure": 0},
                    },
                },
            )

    @app.get("/config", response_class=HTMLResponse)
    async def config_page(request: Request):
        """Configuration editor page."""
        try:
            auth_token = request.cookies.get("auth_token")
            if not auth_token or not verify_auth_token(auth_token):
                logger.warning("Unauthorized access attempt to config page")
                return RedirectResponse(url="/", status_code=302)

            logger.info("Config page accessed successfully")
            return templates.TemplateResponse(
                "config_editor.html", {"request": request}
            )
        except Exception as e:
            logger.error(f"Error accessing config page: {str(e)}")
            raise

    @app.get("/logs", response_class=HTMLResponse)
    async def logs_page(request: Request):
        """Error logs page."""
        try:
            auth_token = request.cookies.get("auth_token")
            if not auth_token or not verify_auth_token(auth_token):
                logger.warning("Unauthorized access attempt to logs page")
                return RedirectResponse(url="/", status_code=302)

            logger.info("Logs page accessed successfully")
            return templates.TemplateResponse("error_logs.html", {"request": request})
        except Exception as e:
            logger.error(f"Error accessing logs page: {str(e)}")
            raise


def setup_health_routes(app: FastAPI) -> None:
    """
    Set up health check-related routes.

    Args:
        app: FastAPI application instance
    """

    @app.get("/health")
    async def health_check(request: Request):
        """Health check endpoint."""
        logger.info("Health check endpoint called")
        return {"status": "healthy"}


def setup_api_stats_routes(app: FastAPI) -> None:
    """
    Set up API statistics-related routes.

    Args:
        app: FastAPI application instance
    """

    @app.get("/api/stats/details")
    async def api_stats_details(request: Request, period: str):
        """Get API call details for a specified period."""
        try:
            auth_token = request.cookies.get("auth_token")
            if not auth_token or not verify_auth_token(auth_token):
                logger.warning("Unauthorized access attempt to API stats details")
                return {"error": "Unauthorized"}, 401

            logger.info(f"Fetching API call details for period: {period}")
            stats_service = StatsService()
            details = await stats_service.get_api_call_details(period)
            return details
        except ValueError as e:
            logger.warning(
                f"Invalid period requested for API stats details: {period} - {str(e)}"
            )
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(
                f"Error fetching API stats details for period {period}: {str(e)}"
            )
            return {"error": "Internal server error"}, 500

    @app.get("/api/stats/attention-keys")
    async def api_stats_attention_keys(
        request: Request, limit: int = 20, status_code: int = 429
    ):
        """Return the keys with the most specified error codes in the last 24 hours (only includes keys in the in-memory list). Default error code is 429."""
        try:
            auth_token = request.cookies.get("auth_token")
            if not auth_token or not verify_auth_token(auth_token):
                logger.warning("Unauthorized access attempt to attention-keys")
                return {"error": "Unauthorized"}, 401

            # Support all standard HTTP status code ranges
            # if not isinstance(status_code, int) or status_code < 100 or status_code > 599:
            #     return {"error": f"Unsupported status_code: {status_code}"}, 400

            if not hasattr(request.app.state, "key_manager"):
                raise RuntimeError("KeyManager not initialized.")
            key_manager = request.app.state.key_manager
            keys_status = await key_manager.get_keys_by_status()
            in_memory_keys = set(keys_status.get("valid_keys", {}).keys()) | set(
                keys_status.get("invalid_keys", {}).keys()
            )
            stats_service = StatsService()
            data = await stats_service.get_attention_keys_last_24h(
                in_memory_keys, limit, status_code
            )
            return data
        except Exception as e:
            logger.error(f"Error fetching attention keys: {e}")
            return {"error": "Internal server error"}, 500

    @app.get("/api/stats/key-details")
    async def api_stats_key_details(request: Request, key: str, period: str):
        """Get call details for a specific key in a specified period."""
        try:
            auth_token = request.cookies.get("auth_token")
            if not auth_token or not verify_auth_token(auth_token):
                logger.warning("Unauthorized access attempt to API key stats details")
                return {"error": "Unauthorized"}, 401

            logger.info(
                f"Fetching key call details for key=...{key[-4:] if key else ''}, period: {period}"
            )
            stats_service = StatsService()
            details = await stats_service.get_key_call_details(key, period)
            return details
        except ValueError as e:
            logger.warning(
                f"Invalid period requested for key stats details: {period} - {str(e)}"
            )
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(
                f"Error fetching key stats details for period {period}: {str(e)}"
            )
            return {"error": "Internal server error"}, 500
