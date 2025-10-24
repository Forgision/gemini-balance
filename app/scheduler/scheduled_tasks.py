from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config.config import settings
from app.domain.gemini_models import GeminiContent, GeminiRequest
from app.log.logger import Logger
from app.service.chat.gemini_chat_service import GeminiChatService
from app.service.error_log.error_log_service import delete_old_error_logs
from app.service.files.files_service import get_files_service
from app.service.key.key_manager import get_key_manager_instance
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
        key_manager = await get_key_manager_instance()
        # Ensure KeyManager is initialized
        if not key_manager or not hasattr(key_manager, "key_failure_counts"):
            logger.warning(
                "KeyManager instance not available or not initialized. Skipping check."
            )
            return

        # Create a GeminiChatService instance for validation
        # Note: An instance is created directly here, not through dependency injection, as this is a background task
        chat_service = GeminiChatService(settings.BASE_URL, key_manager)

        # Get the list of keys to check (failure count > 0)
        keys_to_check = []
        async with (
            key_manager.failure_count_lock
        ):  # Accessing shared data requires a lock
            # Make a copy to avoid modifying the dictionary while iterating
            failure_counts_copy = key_manager.key_failure_counts.copy()
            keys_to_check = [
                key for key, count in failure_counts_copy.items() if count > 0
            ]  # Check all keys with a failure count greater than 0

        if not keys_to_check:
            logger.info("No keys with failure count > 0 found. Skipping verification.")
            return

        logger.info(
            f"Found {len(keys_to_check)} keys with failure count > 0 to verify."
        )

        for key in keys_to_check:
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
                    f"Key {log_key} verification successful. Resetting failure count."
                )
                await key_manager.reset_key_failure_count(key)
            except Exception as e:
                logger.warning(
                    f"Key {log_key} verification failed: {str(e)}. Incrementing failure count."
                )
                # Operate the counter directly, requiring a lock
                async with key_manager.failure_count_lock:
                    # Re-check if the key exists and the failure count has not reached the upper limit
                    if (
                        key in key_manager.key_failure_counts
                        and key_manager.key_failure_counts[key]
                        < key_manager.MAX_FAILURES
                    ):
                        key_manager.key_failure_counts[key] += 1
                        logger.info(
                            f"Failure count for key {log_key} incremented to {key_manager.key_failure_counts[key]}."
                        )
                    elif key in key_manager.key_failure_counts:
                        logger.warning(
                            f"Key {log_key} reached MAX_FAILURES ({key_manager.MAX_FAILURES}). Not incrementing further."
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
