import logging
import asyncio
import itertools
from typing import Optional, Dict, Any
import pandas as pd
import pytz
from datetime import datetime, timedelta
from sqlalchemy import (
    select,
    update as sa_update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from app.log.logger import get_key_manager_logger
from app.service.key.rate_limits import scrape_gemini_rate_limits
from app.config.config import settings
from app.utils.read_write_lock import ReadWriteLock
from app.database.models import UsageMatrix


GEMINI_RATE_LIMIT_URL = (
    "https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits"
)

# Configure logging
logger = get_key_manager_logger()
logger.setLevel(logging.DEBUG)


# TODO: handle model name such as gemini-2.5-flash-search while get max rate limits
class KeyManager:
    """
    Manages API key usage, rate limiting, and automatic resets.

    This class is designed to be async-native and thread-safe for use
    with FastAPI. It must be instantiated using the KeyManager.create()
    classmethod.
    """

    _INDEX_LEVEL = ["model_name", "is_vertex_key", "api_key"]
    _COLUMNS = [
        [
            "rpm",
            "tpm",
            "rpd",
            "max_rpm",
            "max_tpm",
            "max_rpd",
            "is_active",
            "is_exhausted",
            "last_used",
        ]
    ]
    _ALL_COLUMNS = _INDEX_LEVEL + _COLUMNS[0]

    def __init__(
        self,
        api_keys: list[str],
        vertex_api_keys: list[str],
        async_session_maker: async_sessionmaker,
        rate_limit_data: Optional[dict] = None,
        minute_reset_interval: Optional[int] = None,
    ):
        """
        !WARNING: Do not call this directly. Use KeyManager.create()
        """
        # --- Configuration ---
        self.api_keys: list[str] = api_keys
        self.api_keys_cycle = itertools.cycle(api_keys)
        self.vertex_api_keys: list[str] = vertex_api_keys
        self.vertex_api_keys_cycle = itertools.cycle(vertex_api_keys)
        self.db_maker: async_sessionmaker[AsyncSession] = async_session_maker
        self.rate_limit_data: dict = rate_limit_data
        self.rate_limit_models: list[str] = (
            list(rate_limit_data.keys()) if rate_limit_data else []
        )
        self.tz = pytz.timezone(zone="UTC")
        self.now = lambda: datetime.now(self.tz)
        self.now_minute = lambda: self.now().replace(second=0, microsecond=0)
        self.now_day = lambda: self.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.minute_reset_interval_sec = (
            minute_reset_interval or 60
        )  # default 60 seconds

        # --- Core Data ---
        # This DataFrame is the "brain" and holds all state.
        self.df: pd.DataFrame = pd.DataFrame()
        self.lock = ReadWriteLock()

        # --- State Management ---
        self.is_ready = False
        self._background_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._required_db_commit = False

        # --- Timestamps for Resets ---
        self.last_minute_reset_ts: datetime = self.now_minute()
        self.last_day_reset_ts: datetime = self.now_day()

    def _model_normalization(self, model_name: str) -> str:
        """Normalizes a model name by matching it against configured rate limit models.
        This method checks if the input model name matches any prefix in the rate limit
        models list. If a match is found, it returns the longest matching prefix along
        with a success indicator. If no match is found, it returns the original model
        name with a failure indicator.
        Args:
            model_name (str): The model name to normalize. Must be a non-empty string.
        Returns:
            str: The matching prefix if found, otherwise the original model_name
        Raises:
            TypeError: If model_name is not a non-empty string.
            ValueError: If rate_limit_models is not properly configured (None, not a list,
                       or empty).
        Note:
            The method prioritizes longer prefixes when multiple matches exist, ensuring
            the most specific match is returned.

        """
        if not isinstance(model_name, str) or not model_name:
            raise TypeError("Input must be a non-empty string.")

        if (
            self.rate_limit_models is None
            or not isinstance(self.rate_limit_models, list)
            or len(self.rate_limit_models) < 1
        ):
            logger.error("Rate Limits models are not found")
            raise ValueError("Rate Limits models are not found")

        for prefix in sorted(self.rate_limit_models, key=len, reverse=True):
            if model_name.startswith(prefix):
                # As soon as we find a match (which will be the longest one), return it.
                return prefix

        # If the loop finishes without finding any match, return the original string.
        return model_name

    def _get_default_rate_limits(self) -> dict:
        """Return default rate limits if scraping fails."""
        # Return hardcoded defaults (Free Tier structure)
        return {
            "Free Tier": {
                "gemini-pro": {"RPM": 60, "TPM": 1000000, "RPD": 1500},
                "gemini-2.0-flash-exp": {"RPM": 15, "TPM": 1000000, "RPD": 1500},
                "gemini-2.5-pro": {"RPM": 60, "TPM": 1000000, "RPD": 1500},
                "gemini-2.5-flash": {"RPM": 60, "TPM": 1000000, "RPD": 1500},
            }
        }

    async def _check_ready(self):
        error_msg = ""
        if (
            self.rate_limit_data is None
            or not isinstance(self.rate_limit_data, dict)
            or len(self.rate_limit_data) < 1
        ):
            error_msg += "Rate Limits data is not found"

        if self.df is None or not isinstance(self.df, pd.DataFrame):
            error_msg += "\nDataFrame not initialized."
        else:
            missing_columns = [
                column
                for column in [
                    "rpm",
                    "tpm",
                    "rpd",
                    "max_rpm",
                    "max_tpm",
                    "max_rpd",
                    "rpm_left",
                    "tpm_left",
                    "rpd_left",
                ]
                if column not in self.df.columns
            ]
            if missing_columns:
                error_msg += f"\nRequired columns {missing_columns} not found"

        if self.df.empty:
            error_msg += "\nDataFrame is empty."

        if error_msg:
            raise Exception(error_msg)

        return True

    async def _load_default(self):
        """
        Initializes the internal DataFrame (`self.df`) with default API key usage data.

        This method populates the DataFrame with all configured API keys (Gemini and Vertex)
        for each model specified in `self.rate_limit_data`, setting initial usage metrics
        to zero and marking keys as not exhausted. It also sets initial reset times.

        Raises:
            ValueError: If `self.rate_limit_data` is empty or if no API keys
                        (either `self.api_keys` or `self.vertex_api_keys`) are found.
            ValueError: If the number of default entries created does not match the
                        total number of API keys, indicating a potential issue in setup.
        """
        logger.debug("Loading default matrix")

        # Checking rate limit data availability
        if (
            self.rate_limit_data is None
            or not isinstance(self.rate_limit_data, dict)
            or len(self.rate_limit_data) < 1
        ):
            raise ValueError("Rate limits data not found")

        # Checking api keys availability
        if len(self.api_keys) < 1 and len(self.vertex_api_keys) < 1:
            raise ValueError("No api keys found")

        defaults = []

        # Creating list of dictionary from rate limit data which can be passed to DataFrame
        for model, limits in self.rate_limit_data.items():
            # Add defaults gemini api keys
            for key in self.api_keys:
                # adding normal key
                defaults.append(
                    {
                        "api_key": key,
                        "model_name": model,
                        "rpm": 0,
                        "max_rpm": limits["RPM"],
                        "tpm": 0,
                        "max_tpm": limits["TPM"],
                        "rpd": 0,
                        "max_rpd": limits["RPD"],
                        "minute_reset_time": self.now_minute(),
                        "day_reset_time": self.now_day(),
                        "last_used": self.now()
                        - timedelta(2),  # set previous second day as last used
                        "is_vertex_key": False,
                        "is_active": True,
                        "is_exhausted": False,
                    }
                )

            # add vertext api keys
            if len(self.vertex_api_keys) > 0:
                for key in self.vertex_api_keys:
                    defaults.append(
                        {
                            "api_key": key,
                            "model_name": model,
                            "rpm": 0,
                            "max_rpm": limits["RPM"],
                            "tpm": 0,
                            "max_tpm": limits["TPM"],
                            "rpd": 0,
                            "max_rpd": limits["RPD"],
                            "minute_reset_time": self.now_minute(),
                            "day_reset_time": self.now_day(),
                            "last_used": self.now()
                            - timedelta(2),  # set previous second day as last used
                            "is_vertex_key": True,
                            "is_active": True,
                            "is_exhausted": False,
                        }
                    )

        # Checking that all api keys have entries for all models in data
        expected_entries = (len(self.api_keys) + len(self.vertex_api_keys)) * len(
            self.rate_limit_data
        )
        if len(defaults) != expected_entries:
            raise ValueError(
                f"Defaults is not created for each keys. Expected {expected_entries} entries but got {len(defaults)}"
            )

        df = pd.DataFrame(defaults)
        df.set_index(self._INDEX_LEVEL, inplace=True, drop=True)
        df.sort_index(inplace=True)
        # Updating dataframe
        async with self.lock.write_lock():
            self.df = df
            self.last_minute_reset_ts = self.now_minute()
            self.last_day_reset_ts = self.now_day()

    async def _load_from_db(self):
        """
        Loads all key/model configurations from the DB and merges
        them with the provided rate_limit_data.

        This populates self.key_model_metrics_df
        """
        logger.info("Loading key and usage data from database...")

        try:
            # Load data from database
            async with self.db_maker() as session:
                stmt = select(UsageMatrix)
                result = await session.execute(stmt)
                db_data = result.scalars().fetchall()

            if db_data:
                # Convert database data to dictionary
                db_data_dict = [
                    {
                        "api_key": item.api_key,
                        "model_name": item.model_name,
                        "rpm": item.rpm,
                        "rpd": item.rpd,
                        "tpm": item.tpm,
                        "total_token_count": item.total_token_count,
                        "minute_reset_time": item.minute_reset_time,
                        "day_reset_time": item.day_reset_time,
                        "last_used": item.last_used,
                        "is_vertex_key": item.vertex_key,
                        "is_active": item.is_active,
                        "is_exhausted": item.is_exhausted,
                    }
                    for item in db_data
                ]

                df = pd.DataFrame(db_data_dict)
                df.set_index(self._INDEX_LEVEL, inplace=True, drop=True)

                # Ensure datetime columns in df match the dtype in self.df to avoid FutureWarning
                datetime_columns = ["minute_reset_time", "day_reset_time", "last_used"]
                for col in datetime_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], utc=True)

                async with self.lock.write_lock():
                    # self.df is guaranteed to be initialized by _load_default before this method is called.
                    # Ensure self.df is also indexed for proper update.
                    if isinstance(self.df, pd.DataFrame):
                        # Update existing entries in self.df with values from df.
                        self.df.update(df)
                        logger.info(
                            f"Successfully merged {len(df)} records from database into self.df."
                        )
                    else:
                        raise ValueError(
                            "self.df is not a DataFrame. run _load_default first."
                        )

                    # setting reset times

                    # set last minute reset time
                    max_minute_reset_dt = pd.to_datetime(
                        self.df["minute_reset_time"]
                    ).max()
                    if pd.isna(max_minute_reset_dt) or isinstance(
                        max_minute_reset_dt, pd.Series
                    ):
                        self.last_minute_reset_ts = datetime.now(self.tz)
                    else:
                        dt = (
                            max_minute_reset_dt.to_pydatetime()
                            if hasattr(max_minute_reset_dt, "to_pydatetime")
                            else max_minute_reset_dt
                        )
                        # Ensure timezone-aware datetime (assume UTC if naive)
                        if dt.tzinfo is None:
                            dt = self.tz.localize(dt)
                        self.last_minute_reset_ts = dt

                    # set last day reset time
                    max_day_reset_dt = pd.to_datetime(self.df["day_reset_time"]).max()
                    if pd.isna(max_day_reset_dt) or isinstance(
                        max_day_reset_dt, pd.Series
                    ):
                        self.last_day_reset_ts = datetime.now(self.tz)
                    else:
                        dt = (
                            max_day_reset_dt.to_pydatetime()
                            if hasattr(max_day_reset_dt, "to_pydatetime")
                            else max_day_reset_dt
                        )
                        # Ensure timezone-aware datetime (assume UTC if naive)
                        if dt.tzinfo is None:
                            dt = self.tz.localize(dt)
                        self.last_day_reset_ts = dt

            else:
                # No data in DB, self.df remains as initialized by _load_default.
                logger.warning(
                    "No data found in 't_usage_stats'. Using defaults from _load_default."
                )

        except Exception as e:
            logger.error(f"Exception to load from database: {e}", exc_info=True)

    async def _set_available_usage(self):
        """Set available usage for keys based on usage. This should be called by _on_update_usage only."""

        missing_columns = [
            column
            for column in ["rpm", "tpm", "rpd", "max_rpm", "max_tpm", "max_rpd"]
            if column not in self.df.columns
        ]
        if missing_columns:
            raise TypeError(f"Required columns {missing_columns} not found")

        self.df["rpm_left"] = (self.df["max_rpm"] - self.df["rpm"]).clip(lower=0)
        self.df["tpm_left"] = (self.df["max_tpm"] - self.df["tpm"]).clip(lower=0)
        self.df["rpd_left"] = (self.df["max_rpd"] - self.df["rpd"]).clip(lower=0)

    async def _set_exhausted_flags(self):
        """Set exhausted flags for keys based on usage. This should be called by _on_update_usage only."""
        missing_columns = [
            column
            for column in ["rpm", "tpm", "rpd", "max_rpm", "max_tpm", "max_rpd"]
            if column not in self.df.columns
        ]
        if missing_columns:
            raise TypeError(f"Required columns {missing_columns} not found")

        flags = (
            (self.df["rpm"] >= self.df["max_rpm"])
            | (self.df["tpm"] >= self.df["max_tpm"])
            | (self.df["rpd"] >= self.df["max_rpd"])
        )
        if "is_exhausted" in self.df.columns:
            self.df["is_exhausted"] = self.df["is_exhausted"] | flags
        else:
            self.df["is_exhausted"] = flags

    async def _ensure_numeric_columns(self):
        """Ensures that the columns used for comparison are numeric.
        This method should be called within a lock context.
        """
        # This method modifies self.df, so it should be called within a lock
        numeric_cols = ["rpm", "tpm", "rpd", "max_rpm", "max_tpm", "max_rpd"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = (
                    pd.Series(
                        pd.to_numeric(self.df[col], errors="coerce"),
                        index=self.df.index,
                    )
                    .fillna(0)
                    .astype(int)
                )

    async def _on_update_usage(self):
        """This method is called upon any updation in self.df, but must be called within a lock context."""
        await self._set_available_usage()
        await self._set_exhausted_flags()

        # set required db commit
        self._required_db_commit = True

    async def init(
        self,
        async_session_maker: Optional[async_sessionmaker] = None,
        rate_limit_data: Optional[dict] = None,
        minute_reset_interval: Optional[int] = None,
    ) -> bool:
        """
        Async factory for creating and initializing the KeyManager.
        This runs the initial data load and starts background tasks.
        """
        try:
            logger.info("Initializing KeyManager...")

            self.db_maker = async_session_maker or self.db_maker
            self.rate_limit_data: dict = rate_limit_data or self.rate_limit_data
            self.minute_reset_interval_sec = (
                minute_reset_interval or self.minute_reset_interval_sec
            )

            if not self.rate_limit_data:
                try:
                    scraped = scrape_gemini_rate_limits(GEMINI_RATE_LIMIT_URL)
                except Exception as e:
                    logger.warning(
                        f"Failed to scrape rate limits: {e}. Using cached/default."
                    )
                    scraped = None
                if not scraped:
                    scraped = self._get_default_rate_limits()
                if isinstance(scraped, dict) and "Free Tier" in scraped:
                    self.rate_limit_data: dict = scraped["Free Tier"].copy()
                else:
                    self.rate_limit_data: dict = (
                        scraped.copy() if isinstance(scraped, dict) else {}
                    )
            self.rate_limit_models = sorted(
                list(self.rate_limit_data.keys()), key=len, reverse=True
            )

            # load defaults
            await self._load_default()

            # 1. Load data from the database
            await self._load_from_db()

            await self._ensure_numeric_columns()

            # 3. Start background resetter
            self._background_task = asyncio.create_task(self.backgroud_worker())

            # 2. Run initial reset to set timestamps
            async with self.lock.write_lock():
                await self._on_update_usage()

            # 4. Set ready state
            if await self._check_ready():
                self.is_ready = True

            logger.info("KeyManager is ready.")
            return True
        except Exception as e:
            logger.error("KeyManager initialization is failed", exc_info=True)
            # Raise exception instead of returning False
            raise RuntimeError(f"KeyManager initialization failed: {e}") from e

    async def shutdown(self):
        """
        Gracefully shuts down the KeyManager.
        This method is idempotent and can be called multiple times safely.
        """
        # Check if already shutting down or shut down
        if self._stop_event.is_set() and (
            not self._background_task or self._background_task.done()
        ):
            logger.debug("KeyManager already shut down or shutting down.")
            return

        logger.info("Shutting down KeyManager...")
        # 1. Signal background task to stop
        self._stop_event.set()

        # 2. Wait for task to finish (with timeout to prevent hanging)
        # 2. Wait for task to finish (with timeout to prevent hanging)
        if self._background_task and not self._background_task.done():
            # Check if task is in the same loop
            try:
                task_loop = self._background_task.get_loop()
                current_loop = asyncio.get_running_loop()
                if task_loop is not current_loop:
                    logger.warning(
                        "Background task loop differs from current loop. Skipping await."
                    )
                    if not self._background_task.cancelled():
                        task_loop.call_soon_threadsafe(self._background_task.cancel)
                    return
            except Exception:
                # Fallback if get_loop fails or other issues
                pass

            try:
                await asyncio.wait_for(self._background_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Background task did not stop in time, cancelling...")
                self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        # 3. Perform one final commit to the DB (only if initialized)
        if self.is_ready:
            logger.info("Performing final database commit...")
            try:
                await self._commit_to_db()
            except Exception:
                logger.error("Error during final commit", exc_info=True)

        self.is_ready = False
        logger.info("KeyManager shutdown complete.")

    async def get_key(self, model_name: str, is_vertex_key: bool = False) -> str:
        """Get key for a given model name and type (Gemini or Vertex) which has highest tokens left.

        The method ensures that the KeyManager is ready, normalizes the model name, and then acquires
        a lock to safely access key usage records. It filters available keys for the specified model
        and type, checks activity and depletion status, applies current model and global quota limits, and returns the best key with the most tokens left for assignment.

        Args:
            model_name (str): The name of the model for which to fetch an API key.
            is_vertex_key (bool, optional): If True, retrieves a Vertex API key. Otherwise, retrieves a Gemini key. Defaults to False.

        Returns:
            str: The selected API key as a string.

        Raises:
            Exception: If no keys are configured or available for the requested model or key type, or if the internal DataFrame is not ready.
        """

        # Ensure readiness
        if not self.is_ready:
            await self._check_ready()

        # Normalize model name and extract the normalized string
        model_name = self._model_normalization(model_name)

        # Acquire lock early to safely check and access self.df
        async with self.lock.read_lock():
            # Validate DataFrame and that 'model_name' exists at index level 0
            if model_name not in self.df.index.get_level_values("model_name"):
                logger.warning(
                    f"No keys configured for model: {model_name}, falling back to cycle."
                )
                # return next key in cycle, model will be inserted in next update_usage call
                if is_vertex_key:
                    return next(self.vertex_api_keys_cycle)
                else:
                    return next(self.api_keys_cycle)

            # Filter for the specific model and vertex key type
            try:
                model_df = self.df.xs(model_name, level="model_name", drop_level=False)
                candidates = model_df.xs(
                    is_vertex_key, level="is_vertex_key", drop_level=False
                ).copy()
            except KeyError:
                logger.warning(
                    f"No keys configured for model: {model_name}, falling back to cycle."
                )
                return await self.get_next_key(is_vertex_key=is_vertex_key)

        # Apply filters
        mask = (
            (candidates["is_active"])
            & (~candidates["is_exhausted"])
            & (candidates["rpm_left"] >= 1)
            & (candidates["tpm_left"] >= 1)
            & (candidates["rpd_left"] >= 1)
        )

        candidates = candidates.loc[mask]

        # Check data is not empty
        if len(candidates) == 0:
            logger.warning(
                f"No available keys for model {model_name}, falling back to cycle."
            )
            # TODO: raise NoKeyError and handle it in the caller return 429 error HttpException
            return ""  # Return empty string to indicate no key available

        # Sort by tpm_left descending
        candidates.sort_values(by=["tpm_left"], ascending=False, inplace=True)

        # Check index level and get the best key string
        first_index = candidates.index[0]

        if isinstance(first_index, tuple) or isinstance(first_index, list):
            if len(first_index) == 3:
                best_key_string = str(first_index[2])
            elif len(first_index) == 2:
                best_key_string = str(first_index[1])
            elif len(first_index) == 1:
                best_key_string = str(first_index[0])
            else:
                logger.warning(
                    f"Invalid index length: {len(first_index)}, falling back to cycle."
                )
                return await self.get_next_key(is_vertex_key=is_vertex_key)
        elif isinstance(first_index, str):
            best_key_string = first_index
        else:
            logger.warning(
                f"Invalid index type: {type(first_index)}, falling back to cycle."
            )
            return await self.get_next_key(is_vertex_key=is_vertex_key)

        return best_key_string

    async def update_usage(
        self,
        model_name: str,
        key_value: str,
        is_vertex_key: bool,
        tokens_used: int,
        error: bool = False,
        error_type: str | None = None,
    ):
        """
        Update usage statistics for an API key.

        This method updates the in-memory DataFrame usage statistics for the given API key, model name, and token usage.
        The update operation is performed asynchronously and only affects the in-memory cache (no database writes).
        It increments usage counters such as rpd (requests per day), rpm (requests per minute), tpm (tokens per minute),
        and updates fields like last_used.

        If an error occurred, the key may be marked as inactive (for permanent errors) or exhausted (for HTTP 429 errors).

        Args:
            model_name (str): The model name the key was used for.
            key_value (str): The API key string.
            is_vertex_key (bool): Whether the key is for Vertex.
            tokens_used (int): Number of tokens used.
            error (bool, optional): Whether an error occurred.
            error_type (Optional[str], optional): The type of error if any (e.g., "429" or "permanent").
        """

        # convert to int and handle NaN
        def to_int_safe(val) -> int:
            if isinstance(val, (int, float)):
                return int(val) if not pd.isna(val) else 0
            try:
                scalar = val.item() if hasattr(val, "item") else float(val)
                return int(scalar) if not pd.isna(scalar) else 0
            except (ValueError, AttributeError, TypeError):
                return 0

        if not self.is_ready:
            await self._check_ready()

        model_name = self._model_normalization(model_name)
        idx = (model_name, is_vertex_key, key_value)

        # Pre-compute values outside lock
        now = self.now()

        try:
            async with self.lock.write_lock():
                if idx in self.df.index:
                    # Fast atomic update - direct write
                    row = self.df.loc[idx]

                    # Update individual columns (preserves all other columns)
                    self.df.loc[idx, "rpd"] = to_int_safe(row["rpd"]) + 1
                    self.df.loc[idx, "rpm"] = to_int_safe(row["rpm"]) + 1
                    self.df.loc[idx, "tpm"] = to_int_safe(row["tpm"]) + tokens_used
                    self.df.loc[idx, "total_token_count"] = (
                        to_int_safe(row.get("total_token_count", 0)) + tokens_used
                    )
                    self.df.loc[idx, "last_used"] = now

                    if error:
                        if error_type == "permanent":
                            key_mask = (
                                self.df.index.get_level_values("api_key") == key_value
                            )
                            self.df.loc[key_mask, "is_active"] = False
                        elif error_type == "429":
                            self.df.loc[idx, "is_exhausted"] = True

                    # Call _on_update_usage to update usage
                    await self._on_update_usage()
                else:
                    new_entry = {
                        "api_key": key_value,
                        "model_name": model_name,
                        "rpm": 1,
                        "max_rpm": 99999,  # Unlimited
                        "tpm": tokens_used,
                        "max_tpm": 999999999,  # Unlimited
                        "rpd": 1,
                        "max_rpd": 99999,  # Unlimited
                        "total_token_count": tokens_used,
                        "minute_reset_time": self.now_minute(),
                        "day_reset_time": self.now_day(),
                        "last_used": self.now() - timedelta(2),
                        "is_vertex_key": is_vertex_key,
                        "is_active": True
                        if not error or error_type != "permanent"
                        else False,
                        "is_exhausted": True
                        if error and error_type == "429"
                        else False,
                    }
                    new_df = pd.DataFrame([new_entry])
                    new_df.set_index(self._INDEX_LEVEL, inplace=True, drop=True)
                    self.df = pd.concat([self.df, new_df]).sort_index()
                    logger.info(f"Created entry for unknown model: {model_name}")

                    # Call _on_update_usage to update usage
                    await self._on_update_usage()
        except Exception as e:
            logger.error(f"Error in update_usage: {e}", exc_info=True)
            return False

        return True

    async def reset_usage(self):
        """
        The core LOGIC for resetting metrics in the DataFrame.
        This method is called by the background task.
        """
        try:
            if not self.is_ready:
                await self._check_ready()

            now_minute = self.now_minute()
            now_day = self.now_day()

            # Ensure last_minute_reset_ts is timezone-aware (assume UTC if naive)
            if self.last_minute_reset_ts.tzinfo is None:
                self.last_minute_reset_ts = self.tz.localize(self.last_minute_reset_ts)

            if self.last_minute_reset_ts + timedelta(seconds=59) < now_minute:
                logger.debug("Resetting minute-level metrics (rpm, tpm, exhausted)...")
                async with self.lock.write_lock():
                    self.df["rpm"] = 0
                    self.df["tpm"] = 0
                    self.df["is_exhausted"] = False
                    self.last_minute_reset_ts = now_minute
                    await self._on_update_usage()

            # Ensure last_day_reset_ts is timezone-aware (assume UTC if naive)
            if self.last_day_reset_ts.tzinfo is None:
                self.last_day_reset_ts = self.tz.localize(self.last_day_reset_ts)

            if self.last_day_reset_ts + timedelta(days=1) < now_day:
                logger.debug("Resetting day-level metrics (rpd)...")
                async with self.lock.write_lock():
                    self.df["rpd"] = 0
                    self.last_day_reset_ts = now_day
                    await self._on_update_usage()
            return True
        except Exception:
            logger.error("Exception in reset_usage", exc_info=True)
            return False

    async def _commit_to_db(self):
        """
        Saves the current state (rpd, is_active) to the database.
        """
        if self.df is None or self.df.empty:
            logger.debug("Skipping commit: DataFrame is None or empty.")
            return

        # 1. Get a thread-safe copy of the data to save
        async with self.lock.write_lock():
            df_to_save = self.df.copy()

        # 2. Iterate and save
        logger.info(f"Committing {len(df_to_save)} records to database...")
        try:
            async with self.db_maker() as session:
                bind = session.get_bind()
                is_sqlite = (
                    hasattr(bind, "dialect")
                    and getattr(bind.dialect, "name", "") == "sqlite"
                )
                if is_sqlite:
                    for (
                        model_name,
                        is_vertex_key,
                        api_key,
                    ), row in df_to_save.iterrows():  # type: ignore
                        values = {
                            "api_key": api_key,
                            "model_name": model_name,
                            "rpm": int(row["rpm"]),
                            "tpm": int(row["tpm"]),
                            "rpd": int(row["rpd"]),
                            "total_token_count": int(row.get("total_token_count", 0)),
                            "minute_reset_time": self.last_minute_reset_ts,
                            "day_reset_time": self.last_day_reset_ts,
                            "vertex_key": bool(is_vertex_key),
                            "is_active": bool(row["is_active"]),
                            "is_exhausted": bool(row["is_exhausted"]),
                            "last_used": row["last_used"],
                        }
                        stmt = sqlite_insert(UsageMatrix).values(**values)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=[
                                UsageMatrix.api_key,
                                UsageMatrix.model_name,
                                UsageMatrix.vertex_key,
                            ],
                            set_=values,
                        )
                        await session.execute(stmt)
                    await session.commit()
                else:
                    for (
                        model_name,
                        is_vertex_key,
                        api_key,
                    ), row in df_to_save.iterrows():  # type: ignore
                        values = {
                            "rpm": int(row["rpm"]),
                            "tpm": int(row["tpm"]),
                            "rpd": int(row["rpd"]),
                            "total_token_count": int(row.get("total_token_count", 0)),
                            "minute_reset_time": self.last_minute_reset_ts,
                            "day_reset_time": self.last_day_reset_ts,
                            "is_exhausted": bool(row["is_exhausted"]),
                            "last_used": row["last_used"],
                            "is_active": bool(row["is_active"]),
                        }
                        upd = (
                            sa_update(UsageMatrix)
                            .where(
                                (UsageMatrix.api_key == api_key)
                                & (UsageMatrix.model_name == model_name)
                                & (UsageMatrix.vertex_key == bool(is_vertex_key))
                            )
                            .values(**values)
                        )
                        result = await session.execute(upd)
                        if result.rowcount == 0:
                            instance = UsageMatrix(
                                api_key=api_key,
                                model_name=model_name,
                                rpm=values["rpm"],
                                tpm=values["tpm"],
                                rpd=values["rpd"],
                                total_token_count=values["total_token_count"],
                                minute_reset_time=values["minute_reset_time"],
                                day_reset_time=values["day_reset_time"],
                                vertex_key=bool(is_vertex_key),
                                is_active=values["is_active"],
                                is_exhausted=values["is_exhausted"],
                                last_used=values["last_used"],
                            )
                            session.add(instance)
                    await session.commit()
            logger.debug("Database commit successful.")
        except Exception:
            logger.error("Database commit failed", exc_info=True)

    async def backgroud_worker(self):
        """
        The main background task that runs the reset and commit loops.
        """
        logger.info("KeyManager backgroud worker started.")
        while not self._stop_event.is_set():
            try:
                if not self.is_ready:
                    await asyncio.sleep(1)
                    await self._check_ready()

                await self.reset_usage()

                # 3. Check for DB Commit
                if self._required_db_commit:
                    await self._commit_to_db()
                    self._required_db_commit = False

                # Wait for 1 second or until stop_event is set
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass  # This is the normal loop

            except asyncio.CancelledError:
                logger.info("KeyManager backgroud worker cancelled.")
                break
            except Exception:
                logger.error("Exception in KeyManager backgroud worker", exc_info=True)
                # Don't crash the loop, just wait and retry
                await asyncio.sleep(5)

        logger.info("KeyManager backgroud worker stopped.")

    # ========== Adapter Methods for v1 Compatibility ==========

    async def get_paid_key(self) -> str:
        """Get the paid API key for premium features."""
        return settings.PAID_KEY

    async def get_next_key(self, is_vertex_key: bool = False) -> str:
        """Get the next API key in round-robin fashion."""
        try:
            if is_vertex_key:
                return next(self.vertex_api_keys_cycle)
            else:
                return next(self.api_keys_cycle)
        except StopIteration:
            logger.warning("API key cycle is empty.")
            return ""

    async def get_random_valid_key(self, model_name: Optional[str] = None) -> str:
        """
        Adapter method for v1 compatibility.
        Returns a key from cycle (round-robin).
        """
        return await self.get_next_key(is_vertex_key=False)

    async def get_usage_stats(
        self, api_key: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get usage statistics for a given key and model.
        Adapter method for v1 compatibility.
        """
        if not self.is_ready:
            await self._check_ready()

        model_name = self._model_normalization(model_name)
        is_vertex_key = api_key in self.vertex_api_keys

        async with self.lock.read_lock():
            try:
                idx = (model_name, is_vertex_key, api_key)
                if idx not in self.df.index:
                    return None

                row = self.df.loc[idx].to_dict()  # type: ignore[index]
                return {
                    "rpm": int(row["rpm"]),
                    "tpm": int(row["tpm"]),
                    "rpd": int(row["rpd"]),
                    "max_rpm": int(row["max_rpm"]),
                    "max_tpm": int(row["max_tpm"]),
                    "max_rpd": int(row["max_rpd"]),
                    "rpm_left": int(row.get("rpm_left", 0)),
                    "tpm_left": int(row.get("tpm_left", 0)),
                    "rpd_left": int(row.get("rpd_left", 0)),
                    "is_active": bool(row["is_active"]),
                    "is_exhausted": bool(row["is_exhausted"]),
                    "last_used": row["last_used"],
                    "minute_reset_time": row.get("minute_reset_time"),
                    "day_reset_time": row.get("day_reset_time"),
                }
            except (KeyError, IndexError):
                return None

    async def get_keys_by_status(self) -> dict:
        """
        Returns keys grouped by status in v2 format.
        Uses is_active and is_exhausted flags from DataFrame.
        """
        if not self.is_ready:
            await self._check_ready()

        valid_keys = {}
        invalid_keys = {}

        async with self.lock.read_lock():
            # Get all unique keys from DataFrame
            all_keys = set(self.df.index.get_level_values("api_key"))

            for key in all_keys:
                # Check if key is active for any model
                key_rows = self.df[self.df.index.get_level_values("api_key") == key]
                is_any_active = (
                    key_rows["is_active"].any() if not key_rows.empty else False
                )
                is_any_exhausted = (
                    key_rows["is_exhausted"].any() if not key_rows.empty else False
                )

                # v2 format: use status flags
                if is_any_active and not is_any_exhausted:
                    valid_keys[key] = {"status": "active", "exhausted": False}
                elif is_any_exhausted:
                    valid_keys[key] = {"status": "exhausted", "exhausted": True}
                else:
                    invalid_keys[key] = {"status": "inactive", "exhausted": False}

        return {"valid_keys": valid_keys, "invalid_keys": invalid_keys}

    async def get_all_keys_with_fail_count(self) -> dict:
        """
        Adapter method for v1 compatibility.
        Returns all keys with status. Since v2 doesn't track failure counts,
        uses is_active flag (0 = valid, 1 = invalid).
        """
        status = await self.get_keys_by_status()
        all_keys = {**status["valid_keys"], **status["invalid_keys"]}

        return {
            "valid_keys": status["valid_keys"],
            "invalid_keys": status["invalid_keys"],
            "all_keys": all_keys,
        }

    async def reset_key_failure_count(self, key: str) -> bool:
        """
        Manually reactivate a key (adapter for v1 compatibility).
        Reactivates for all models.
        Note: Auto-reset will handle periodic resets, but this allows immediate reactivation.
        """
        if not self.is_ready:
            await self._check_ready()

        from app.utils.helpers import redact_key_for_logging

        async with self.lock.write_lock():
            try:
                key_mask = self.df.index.get_level_values("api_key") == key
                if key_mask.any():
                    self.df.loc[key_mask, "is_active"] = True
                    self.df.loc[key_mask, "is_exhausted"] = False
                    await self._on_update_usage()
                    return True
                else:
                    logger.warning(
                        f"Key not found for reactivation: {redact_key_for_logging(key)}"
                    )
                    return False
            except Exception as e:
                logger.error(f"Error reactivating key: {e}", exc_info=True)
                return False

        logger.info(f"Manually reactivated key: {redact_key_for_logging(key)}")
        return False

    async def handle_api_failure(
        self,
        api_key: str,
        model_name: str,
        retries: int,
        is_vertex_key: Optional[bool] = None,
        status_code: Optional[int] = None,
    ) -> str:
        """
        Handle API call failure. Compatible with v1 signature.
        If status_code not provided, defaults to 'permanent' error.
        """
        # Auto-detect vertex key if not provided
        if is_vertex_key is None:
            is_vertex_key = api_key in self.vertex_api_keys

        # Determine error type (default to permanent if status_code not provided)
        if status_code == 429:
            error_type = "429"
        else:
            error_type = "permanent"

        # Update usage (marks key as exhausted/inactive)
        await self.update_usage(
            model_name=model_name,
            key_value=api_key,
            is_vertex_key=is_vertex_key,
            tokens_used=0,
            error=True,
            error_type=error_type,
        )

        # Return new key for retry (match v1 behavior)
        if retries < settings.MAX_RETRIES:
            # get_key() now returns empty string instead of raising (see Phase 2.1)
            return await self.get_key(
                model_name or "gemini-pro", is_vertex_key=is_vertex_key
            )
        return ""

    def _format_key_info(self, api_key, row) -> dict:
        """Helper to format key information."""
        return {
            "api_key": api_key,
            "rpm": int(row["rpm"]),
            "tpm": int(row["tpm"]),
            "rpd": int(row["rpd"]),
            "max_rpm": int(row["max_rpm"]),
            "max_tpm": int(row["max_tpm"]),
            "max_rpd": int(row["max_rpd"]),
            "rpm_left": int(row.get("rpm_left", 0)),
            "tpm_left": int(row.get("tpm_left", 0)),
            "rpd_left": int(row.get("rpd_left", 0)),
            "last_used": row["last_used"].isoformat()
            if pd.notna(row["last_used"])
            else None,
        }

    async def get_state(self) -> dict:
        """
        Get comprehensive state of all keys for monitoring.
        Returns all models (UI handles display of 3 initially).
        """
        if not self.is_ready:
            await self._check_ready()

        async with self.lock.read_lock():
            df_copy = self.df.copy()

        state = {
            "models": {},
            "summary": {
                "total_keys": len(self.api_keys) + len(self.vertex_api_keys),
                "total_models": len(self.rate_limit_models),
                "total_available": 0,
                "total_exhausted": 0,
                "total_inactive": 0,
            },
        }

        for model in self.rate_limit_models:
            if model not in df_copy.index.get_level_values("model_name"):
                continue

            model_data = {
                "model_name": model,
                "regular_keys": {"available": [], "exhausted": [], "inactive": []},
                "vertex_keys": {"available": [], "exhausted": [], "inactive": []},
            }

            # Process regular keys
            try:
                regular_df = df_copy.xs(model, level="model_name").copy()
                regular_df = regular_df.xs(False, level="is_vertex_key").copy()
                for api_key, row in regular_df.iterrows():
                    key_info = self._format_key_info(api_key, row)
                    if not row["is_active"]:
                        model_data["regular_keys"]["inactive"].append(key_info)
                        state["summary"]["total_inactive"] += 1
                    elif row["is_exhausted"]:
                        model_data["regular_keys"]["exhausted"].append(key_info)
                        state["summary"]["total_exhausted"] += 1
                    else:
                        model_data["regular_keys"]["available"].append(key_info)
                        state["summary"]["total_available"] += 1
            except KeyError:
                pass

            # Process vertex keys
            try:
                vertex_df = df_copy.xs(model, level="model_name").copy()
                vertex_df = vertex_df.xs(True, level="is_vertex_key").copy()
                for api_key, row in vertex_df.iterrows():
                    key_info = self._format_key_info(api_key, row)
                    if not row["is_active"]:
                        model_data["vertex_keys"]["inactive"].append(key_info)
                        state["summary"]["total_inactive"] += 1
                    elif row["is_exhausted"]:
                        model_data["vertex_keys"]["exhausted"].append(key_info)
                        state["summary"]["total_exhausted"] += 1
                    else:
                        model_data["vertex_keys"]["available"].append(key_info)
                        state["summary"]["total_available"] += 1
            except KeyError:
                pass

            state["models"][model] = model_data

        return state
