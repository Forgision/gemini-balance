"""
Application Configuration Module
"""

import datetime
import json
from typing import Any, Dict, List, Type, get_args, get_origin

from pydantic import Field, ValidationError, ValidationInfo, field_validator
from pydantic_settings import BaseSettings
from sqlalchemy import insert, select, update

from app.core.constants import (
    API_VERSION,
    DEFAULT_CREATE_IMAGE_MODEL,
    DEFAULT_FILTER_MODELS,
    DEFAULT_MODEL,
    DEFAULT_SAFETY_SETTINGS,
    DEFAULT_STREAM_CHUNK_SIZE,
    DEFAULT_STREAM_LONG_TEXT_THRESHOLD,
    DEFAULT_STREAM_MAX_DELAY,
    DEFAULT_STREAM_MIN_DELAY,
    DEFAULT_STREAM_SHORT_TEXT_THRESHOLD,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
)
from app.log.logger import Logger


class Settings(BaseSettings):
    # Database Configuration
    DATABASE_TYPE: str = "sqlite"  # sqlite or mysql
    SQLITE_DATABASE: str = "default_db"
    MYSQL_HOST: str = ""
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = ""
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = ""
    MYSQL_SOCKET: str = ""

    # Validate MySQL Configuration
    @field_validator(
        "MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"
    )
    def validate_mysql_config(cls, v: Any, info: ValidationInfo) -> Any:
        if info.data.get("DATABASE_TYPE") == "mysql":
            if v is None or v == "":
                raise ValueError(
                    "MySQL configuration is required when DATABASE_TYPE is 'mysql'"
                )
        return v

    # API Related Configuration
    API_KEYS: List[str] = []
    ALLOWED_TOKENS: List[str] = []
    BASE_URL: str = f"https://generativelanguage.googleapis.com/{API_VERSION}"
    AUTH_TOKEN: str = ""
    MAX_FAILURES: int = 3
    TEST_MODEL: str = DEFAULT_MODEL
    TIME_OUT: int = DEFAULT_TIMEOUT
    MAX_RETRIES: int = MAX_RETRIES
    PROXIES: List[str] = []
    PROXIES_USE_CONSISTENCY_HASH_BY_API_KEY: bool = (
        True  # Whether to use consistent hashing to select a proxy
    )
    VERTEX_API_KEYS: List[str] = []
    VERTEX_EXPRESS_BASE_URL: str = (
        "https://aiplatform.googleapis.com/v1beta1/publishers/google"
    )

    # Smart Routing Configuration
    URL_NORMALIZATION_ENABLED: bool = False  # Whether to enable smart routing mapping

    # Custom Headers
    CUSTOM_HEADERS: Dict[str, str] = {}

    # Model Related Configuration
    SEARCH_MODELS: List[str] = ["gemini-2.5-flash", "gemini-2.5-pro"]
    IMAGE_MODELS: List[str] = ["gemini-2.0-flash-exp", "gemini-2.5-flash-image-preview"]
    FILTERED_MODELS: List[str] = DEFAULT_FILTER_MODELS
    TOOLS_CODE_EXECUTION_ENABLED: bool = False
    # Whether to enable URL context
    URL_CONTEXT_ENABLED: bool = False
    URL_CONTEXT_MODELS: List[str] = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-live-001",
    ]
    SHOW_SEARCH_LINK: bool = True
    SHOW_THINKING_PROCESS: bool = True
    THINKING_MODELS: List[str] = []
    THINKING_BUDGET_MAP: Dict[str, float] = {}

    # TTS Related Configuration
    TTS_MODEL: str = "gemini-2.5-flash-preview-tts"
    TTS_VOICE_NAME: str = "Zephyr"
    TTS_SPEED: str = "normal"

    # Image Generation Related Configuration
    PAID_KEY: str = ""
    CREATE_IMAGE_MODEL: str = DEFAULT_CREATE_IMAGE_MODEL
    UPLOAD_PROVIDER: str = "smms"
    SMMS_SECRET_TOKEN: str = ""
    PICGO_API_KEY: str = ""
    PICGO_API_URL: str = "https://www.picgo.net/api/1/upload"
    CLOUDFLARE_IMGBED_URL: str = ""
    CLOUDFLARE_IMGBED_AUTH_CODE: str = ""
    CLOUDFLARE_IMGBED_UPLOAD_FOLDER: str = ""
    # Alibaba Cloud OSS Configuration
    OSS_ENDPOINT: str = ""
    OSS_ENDPOINT_INNER: str = ""
    OSS_ACCESS_KEY: str = ""
    OSS_ACCESS_KEY_SECRET: str = ""
    OSS_BUCKET_NAME: str = ""
    OSS_REGION: str = ""

    # Streaming Output Optimizer Configuration
    STREAM_OPTIMIZER_ENABLED: bool = False
    STREAM_MIN_DELAY: float = DEFAULT_STREAM_MIN_DELAY
    STREAM_MAX_DELAY: float = DEFAULT_STREAM_MAX_DELAY
    STREAM_SHORT_TEXT_THRESHOLD: int = DEFAULT_STREAM_SHORT_TEXT_THRESHOLD
    STREAM_LONG_TEXT_THRESHOLD: int = DEFAULT_STREAM_LONG_TEXT_THRESHOLD
    STREAM_CHUNK_SIZE: int = DEFAULT_STREAM_CHUNK_SIZE

    # Fake Streaming Configuration
    FAKE_STREAM_ENABLED: bool = False  # Whether to enable fake streaming output
    FAKE_STREAM_EMPTY_DATA_INTERVAL_SECONDS: int = (
        5  # Interval for sending empty data in fake streaming (seconds)
    )

    # Scheduler Configuration
    CHECK_INTERVAL_HOURS: int = 1  # Default check interval is 1 hour
    TIMEZONE: str = "Asia/Shanghai"  # Default timezone

    # Github
    GITHUB_REPO_OWNER: str = "snailyp"
    GITHUB_REPO_NAME: str = "gemini-balance"

    # Log Configuration
    LOG_LEVEL: str = "INFO"
    ERROR_LOG_RECORD_REQUEST_BODY: bool = False
    AUTO_DELETE_ERROR_LOGS_ENABLED: bool = True
    AUTO_DELETE_ERROR_LOGS_DAYS: int = 7
    AUTO_DELETE_REQUEST_LOGS_ENABLED: bool = False
    AUTO_DELETE_REQUEST_LOGS_DAYS: int = 30
    SAFETY_SETTINGS: List[Dict[str, str]] = DEFAULT_SAFETY_SETTINGS

    # Files API
    FILES_CLEANUP_ENABLED: bool = True
    FILES_CLEANUP_INTERVAL_HOURS: int = 1
    FILES_USER_ISOLATION_ENABLED: bool = True

    # Admin Session Configuration
    ADMIN_SESSION_EXPIRE: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Admin session expiration time in seconds (5 minutes to 24 hours)",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set default AUTH_TOKEN (if not provided)
        if not self.AUTH_TOKEN and self.ALLOWED_TOKENS:
            self.AUTH_TOKEN = self.ALLOWED_TOKENS[0]


# Create a global configuration instance
settings = Settings()


def _parse_db_value(key: str, db_value: str, target_type: Type) -> Any:
    """Attempt to parse a database string value into the target Python type"""
    from app.log.logger import get_config_logger

    logger = get_config_logger()
    try:
        origin_type = get_origin(target_type)
        args = get_args(target_type)

        # Handle List type
        if origin_type is list:
            # Handle List[str]
            if args and args[0] is str:
                try:
                    parsed = json.loads(db_value)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except json.JSONDecodeError:
                    return [
                        item.strip() for item in db_value.split(",") if item.strip()
                    ]
                logger.warning(
                    f"Could not parse '{db_value}' as List[str] for key '{key}', falling back to comma split or empty list."
                )
                return [item.strip() for item in db_value.split(",") if item.strip()]
            # Handle List[Dict[str, str]]
            elif args and get_origin(args[0]) is dict:
                try:
                    parsed = json.loads(db_value)
                    if isinstance(parsed, list):
                        valid = all(
                            isinstance(item, dict)
                            and all(isinstance(k, str) for k in item.keys())
                            and all(isinstance(v, str) for v in item.values())
                            for item in parsed
                        )
                        if valid:
                            return parsed
                        else:
                            logger.warning(
                                f"Invalid structure in List[Dict[str, str]] for key '{key}'. Value: {db_value}"
                            )
                            return []
                    else:
                        logger.warning(
                            f"Parsed DB value for key '{key}' is not a list type. Value: {db_value}"
                        )
                        return []
                except json.JSONDecodeError:
                    logger.error(
                        f"Could not parse '{db_value}' as JSON for List[Dict[str, str]] for key '{key}'. Returning empty list."
                    )
                    return []
                except Exception as e:
                    logger.error(
                        f"Error parsing List[Dict[str, str]] for key '{key}': {e}. Value: {db_value}. Returning empty list."
                    )
                    return []
        # Handle Dict type
        elif origin_type is dict:
            # Handle Dict[str, str]
            if args and args == (str, str):
                parsed_dict = {}
                try:
                    parsed = json.loads(db_value)
                    if isinstance(parsed, dict):
                        parsed_dict = {str(k): str(v) for k, v in parsed.items()}
                    else:
                        logger.warning(
                            f"Parsed DB value for key '{key}' is not a dictionary type. Value: {db_value}"
                        )
                except json.JSONDecodeError:
                    logger.error(
                        f"Could not parse '{db_value}' as Dict[str, str] for key '{key}'. Returning empty dict."
                    )
                return parsed_dict
            # Handle Dict[str, float]
            elif args and args == (str, float):
                parsed_dict = {}
                try:
                    parsed = json.loads(db_value)
                    if isinstance(parsed, dict):
                        parsed_dict = {str(k): float(v) for k, v in parsed.items()}
                    else:
                        logger.warning(
                            f"Parsed DB value for key '{key}' is not a dictionary type. Value: {db_value}"
                        )
                except (json.JSONDecodeError, ValueError, TypeError) as e1:
                    if isinstance(e1, json.JSONDecodeError) and "'" in db_value:
                        logger.warning(
                            f"Failed initial JSON parse for key '{key}'. Attempting to replace single quotes. Error: {e1}"
                        )
                        try:
                            corrected_db_value = db_value.replace("'", '"')
                            parsed = json.loads(corrected_db_value)
                            if isinstance(parsed, dict):
                                parsed_dict = {
                                    str(k): float(v) for k, v in parsed.items()
                                }
                            else:
                                logger.warning(
                                    f"Parsed DB value (after quote replacement) for key '{key}' is not a dictionary type. Value: {corrected_db_value}"
                                )
                        except (json.JSONDecodeError, ValueError, TypeError) as e2:
                            logger.error(
                                f"Could not parse '{db_value}' as Dict[str, float] for key '{key}' even after replacing quotes: {e2}. Returning empty dict."
                            )
                    else:
                        logger.error(
                            f"Could not parse '{db_value}' as Dict[str, float] for key '{key}': {e1}. Returning empty dict."
                        )
                return parsed_dict
        # Handle bool
        elif target_type is bool:
            return db_value.lower() in ("true", "1", "yes", "on")
        # Handle int
        elif target_type is int:
            return int(db_value)
        # Handle float
        elif target_type is float:
            return float(db_value)
        # Default to str or other types that pydantic can handle directly
        else:
            return db_value
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        logger.warning(
            f"Failed to parse db_value '{db_value}' for key '{key}' as type {target_type}: {e}. Using original string value."
        )
        return db_value  # Return the original string if parsing fails


async def sync_initial_settings():
    """
    Synchronize settings on application startup:
    1. Load settings from the database.
    2. Merge database settings into memory settings (database takes precedence).
    3. Synchronize the final memory settings back to the database.
    """
    from app.log.logger import get_config_logger

    logger = get_config_logger()
    # Deferred import to avoid circular dependencies and ensure the database connection is initialized
    from app.database.connection import database
    from app.database.models import Settings as SettingsModel

    global settings
    logger.info("Starting initial settings synchronization...")

    if not database.is_connected:
        try:
            await database.connect()
            logger.info("Database connection established for initial sync.")
        except Exception as e:
            logger.error(
                f"Failed to connect to database for initial settings sync: {e}. Skipping sync."
            )
            return

    try:
        # 1. Load settings from the database
        db_settings_raw: List[Dict[str, Any]] = []
        try:
            query = select(SettingsModel.key, SettingsModel.value)
            results = await database.fetch_all(query)
            db_settings_raw = [
                {"key": row["key"], "value": row["value"]} for row in results
            ]
            logger.info(f"Fetched {len(db_settings_raw)} settings from database.")
        except Exception as e:
            logger.error(
                f"Failed to fetch settings from database: {e}. Proceeding with environment/dotenv settings."
            )
            # Continue even if database read fails, to ensure env/dotenv-based config can be synced to the DB

        db_settings_map: Dict[str, str] = {
            s["key"]: s["value"] for s in db_settings_raw
        }

        # 2. Merge database settings into memory settings (database takes precedence)
        updated_in_memory = False

        for key, db_value in db_settings_map.items():
            if key == "DATABASE_TYPE":
                logger.debug(
                    f"Skipping update of '{key}' in memory from database. "
                    "This setting is controlled by environment/dotenv."
                )
                continue
            if hasattr(settings, key):
                target_type = Settings.__annotations__.get(key)
                if target_type:
                    try:
                        parsed_db_value = _parse_db_value(key, db_value, target_type)
                        memory_value = getattr(settings, key)

                        # Compare the parsed value with the value in memory
                        # Note: For complex types like lists, direct comparison may not be robust, but it's simplified here
                        if parsed_db_value != memory_value:
                            # Check if types match, in case the parse function returned an incompatible type
                            type_match = False
                            origin_type = get_origin(target_type)
                            if origin_type:  # It's a generic type
                                if isinstance(parsed_db_value, origin_type):
                                    type_match = True
                            # It's a non-generic type, or a specific generic we want to handle
                            elif isinstance(parsed_db_value, target_type):
                                type_match = True

                            if type_match:
                                setattr(settings, key, parsed_db_value)
                                logger.debug(
                                    f"Updated setting '{key}' in memory from database value ({target_type})."
                                )
                                updated_in_memory = True
                            else:
                                logger.warning(
                                    f"Parsed DB value type mismatch for key '{key}'. Expected {target_type}, got {type(parsed_db_value)}. Skipping update."
                                )

                    except Exception as e:
                        logger.error(
                            f"Error processing database setting for key '{key}': {e}"
                        )
            else:
                logger.warning(
                    f"Database setting '{key}' not found in Settings model definition. Ignoring."
                )

        # If there were updates in memory, re-validate the Pydantic model (optional but recommended)
        if updated_in_memory:
            try:
                # Reload to ensure type conversion and validation
                settings = Settings(**settings.model_dump())
                logger.info(
                    "Settings object re-validated after merging database values."
                )
            except ValidationError as e:
                logger.error(
                    f"Validation error after merging database settings: {e}. Settings might be inconsistent."
                )

        # 3. Synchronize the final memory settings back to the database
        final_memory_settings = settings.model_dump()
        settings_to_update: List[Dict[str, Any]] = []
        settings_to_insert: List[Dict[str, Any]] = []
        now = datetime.datetime.now(datetime.timezone.utc)

        existing_db_keys = set(db_settings_map.keys())

        for key, value in final_memory_settings.items():
            if key == "DATABASE_TYPE":
                logger.debug(
                    f"Skipping synchronization of '{key}' to database. "
                    "This setting is controlled by environment/dotenv."
                )
                continue

            # Serialize values to string or JSON string
            if isinstance(value, (list, dict)):
                db_value = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, bool):
                db_value = str(value).lower()
            elif value is None:
                db_value = ""
            else:
                db_value = str(value)

            data = {
                "key": key,
                "value": db_value,
                "description": f"{key} configuration setting",
                "updated_at": now,
            }

            if key in existing_db_keys:
                # Only update if the value is different from the one in the database
                if db_settings_map[key] != db_value:
                    settings_to_update.append(data)
            else:
                # Insert if the key is not in the database
                data["created_at"] = now
                settings_to_insert.append(data)

        # Execute bulk insert and update in a transaction
        if settings_to_insert or settings_to_update:
            try:
                async with database.transaction():
                    if settings_to_insert:
                        # Get existing descriptions to avoid overwriting
                        query_existing = select(
                            SettingsModel.key, SettingsModel.description
                        ).where(
                            SettingsModel.key.in_(
                                [s["key"] for s in settings_to_insert]
                            )
                        )
                        existing_desc = {
                            row["key"]: row["description"]
                            for row in await database.fetch_all(query_existing)
                        }
                        for item in settings_to_insert:
                            item["description"] = existing_desc.get(
                                item["key"], item["description"]
                            )

                        query_insert = insert(SettingsModel).values(settings_to_insert)
                        await database.execute(query=query_insert)
                        logger.info(
                            f"Synced (inserted) {len(settings_to_insert)} settings to database."
                        )

                    if settings_to_update:
                        # Get existing descriptions to avoid overwriting
                        query_existing = select(
                            SettingsModel.key, SettingsModel.description
                        ).where(
                            SettingsModel.key.in_(
                                [s["key"] for s in settings_to_update]
                            )
                        )
                        existing_desc = {
                            row["key"]: row["description"]
                            for row in await database.fetch_all(query_existing)
                        }

                        for setting_data in settings_to_update:
                            setting_data["description"] = existing_desc.get(
                                setting_data["key"], setting_data["description"]
                            )
                            query_update = (
                                update(SettingsModel)
                                .where(SettingsModel.key == setting_data["key"])
                                .values(
                                    value=setting_data["value"],
                                    description=setting_data["description"],
                                    updated_at=setting_data["updated_at"],
                                )
                            )
                            await database.execute(query=query_update)
                        logger.info(
                            f"Synced (updated) {len(settings_to_update)} settings to database."
                        )
            except Exception as e:
                logger.error(
                    f"Failed to sync settings to database during startup: {str(e)}"
                )
        else:
            logger.info(
                "No setting changes detected between memory and database during initial sync."
            )

        # Refresh log levels
        log_level = final_memory_settings.get("LOG_LEVEL")
        if isinstance(log_level, str):
            Logger.update_log_levels(log_level)

    except Exception as e:
        logger.error(f"An unexpected error occurred during initial settings sync: {e}")
    finally:
        if database.is_connected:
            try:
                pass
            except Exception as e:
                logger.error(f"Error disconnecting database after initial sync: {e}")

    logger.info("Initial settings synchronization finished.")
