from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config.config import settings
from app.domain.gemini_models import GeminiContent, GeminiRequest
from app.log.logger import Logger
from app.service.chat.gemini_chat_service import GeminiChatService
from app.service.error_log.error_log_service import delete_old_error_logs
from app.service.files.files_service import get_files_service
# App reference will be set when scheduler starts
_app_reference = None

def set_app_reference(app):
    """Set app reference for scheduled tasks."""
    global _app_reference
    _app_reference = app
from app.service.request_log.request_log_service import delete_old_request_logs_task
from app.utils.helpers import redact_key_for_logging

logger = Logger.setup_logger("scheduler")


async def check_failed_keys():
    """
    Periodically checks API keys with a failure count greater than 0 and attempts to validate them.
    If validation is successful, the failure count is reset; if it fails, the failure count is incremented.
    """
    logger.info("Starting scheduled check for failed API keys...")
    try:
        global _app_reference
        if not _app_reference or not hasattr(_app_reference.state, "key_manager"):
            logger.warning("App reference not available. Skipping check.")
            return
        key_manager = _app_reference.state.key_manager
        # Ensure KeyManager is initialized
        if not key_manager or not hasattr(key_manager, "reset_key_failure_count"):
            logger.warning(
                "KeyManager instance not available or not initialized. Skipping check."
            )
            return

        # Create a GeminiChatService instance for validation
        # Note: An instance is created directly here, not through dependency injection, as this is a background task
        chat_service = GeminiChatService(settings.BASE_URL, key_manager)

        # Get all keys and verify them (v2 uses reset_key_failure_count for manual reactivation)
        # Note: v2 doesn't track failure counts, so we'll verify all keys and reactivate failed ones
        # For v2 compatibility, we'll just verify keys using get_keys_by_status
        keys_status = await key_manager.get_keys_by_status()
        invalid_keys = list(keys_status.get("invalid_keys", {}).keys())
        
        if not invalid_keys:
            logger.info("No invalid keys found. Skipping verification.")
            return

        logger.info(f"Found {len(invalid_keys)} invalid keys to verify.")

        for key in invalid_keys:
            # Redact part of the key for logging
            log_key = redact_key_for_logging(key)
            logger.info(f"Verifying key: {log_key}...")
            try:
                # Construct a test request
                gemini_request = GeminiRequest(
                    contents=[
                        GeminiContent(
                            role="user",
                            parts=[{"text": "hi"}],
                        )
                    ]
                )
                await chat_service.generate_content(
                    settings.TEST_MODEL, gemini_request, key
                )
                logger.info(
                    f"Key {log_key} verification successful. Reactivating key."
                )
                await key_manager.reset_key_failure_count(key)
            except Exception as e:
                logger.warning(
                    f"Key {log_key} verification failed: {str(e)}."
                )

    except Exception as e:
        logger.error(
            f"An error occurred during the scheduled key check: {str(e)}", exc_info=True
        )


async def cleanup_expired_files():
    """
    Periodically clean up expired file records
    """
    logger.info("Starting scheduled cleanup for expired files...")
    try:
        files_service = await get_files_service()
        deleted_count = await files_service.cleanup_expired_files()

        if deleted_count > 0:
            logger.info(f"Successfully cleaned up {deleted_count} expired files.")
        else:
            logger.info("No expired files to clean up.")

    except Exception as e:
        logger.error(
            f"An error occurred during the scheduled file cleanup: {str(e)}",
            exc_info=True,
        )


def setup_scheduler():
    """Set up and start APScheduler"""
    scheduler = AsyncIOScheduler(
        timezone=str(settings.TIMEZONE)
    )  # Read timezone from configuration
    # Add a scheduled task to check failed keys
    if settings.CHECK_INTERVAL_HOURS != 0:
        scheduler.add_job(
            check_failed_keys,
            "interval",
            hours=settings.CHECK_INTERVAL_HOURS,
            id="check_failed_keys_job",
            name="Check Failed API Keys",
        )
        logger.info(
            f"Key check job scheduled to run every {settings.CHECK_INTERVAL_HOURS} hour(s)."
        )

    # New: Add a scheduled task to automatically delete error logs, executed at midnight every day
    scheduler.add_job(
        delete_old_error_logs,
        "cron",
        hour=0,
        minute=0,
        id="delete_old_error_logs_job",
        name="Delete Old Error Logs",
    )
    logger.info("Auto-delete error logs job scheduled to run daily at 3:00 AM.")

    # New: Add a scheduled task to automatically delete request logs, executed at midnight every day
    scheduler.add_job(
        delete_old_request_logs_task,
        "cron",
        hour=0,
        minute=0,
        id="delete_old_request_logs_job",
        name="Delete Old Request Logs",
    )
    logger.info(
        f"Auto-delete request logs job scheduled to run daily at 3:05 AM, if enabled and AUTO_DELETE_REQUEST_LOGS_DAYS is set to {settings.AUTO_DELETE_REQUEST_LOGS_DAYS} days."
    )

    # New: Add a scheduled task for file expiration cleanup, executed once per hour
    if getattr(settings, "FILES_CLEANUP_ENABLED", True):
        cleanup_interval = getattr(settings, "FILES_CLEANUP_INTERVAL_HOURS", 1)
        scheduler.add_job(
            cleanup_expired_files,
            "interval",
            hours=cleanup_interval,
            id="cleanup_expired_files_job",
            name="Cleanup Expired Files",
        )
        logger.info(
            f"File cleanup job scheduled to run every {cleanup_interval} hour(s)."
        )

    scheduler.start()
    logger.info("Scheduler started with all jobs.")
    return scheduler


# A global scheduler instance can be added here to stop it gracefully when the application is closed
scheduler_instance = None


def start_scheduler():
    global scheduler_instance
    if scheduler_instance is None or not scheduler_instance.running:
        logger.info("Starting scheduler...")
        scheduler_instance = setup_scheduler()
    logger.info("Scheduler is already running.")


def stop_scheduler():
    global scheduler_instance
    if scheduler_instance and scheduler_instance.running:
        scheduler_instance.shutdown()
        logger.info("Scheduler stopped.")
