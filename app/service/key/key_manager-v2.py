import asyncio
import pandas as pd
import pytz
from datetime import datetime, date
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.sql import text
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class KeyManager:
    """
    Manages API key usage, rate limiting, and automatic resets.

    This class is designed to be async-native and thread-safe for use
    with FastAPI. It must be instantiated using the KeyManager.create()
    classmethod.
    """

    def __init__(
        self,
        async_session_maker: async_sessionmaker,
        rate_limit_data: dict,
        commit_interval: int,
        minute_reset_interval: int,
        tz_name: str,
    ):
        """
        !WARNING: Do not call this directly. Use KeyManager.create()
        """
        # --- Configuration ---
        self.db_maker = async_session_maker
        self.rate_limit_data = rate_limit_data
        self.tz = pytz.timezone(tz_name)
        self.commit_interval_sec = commit_interval
        self.minute_reset_interval_sec = minute_reset_interval

        # --- Core Data ---
        # This DataFrame is the "brain" and holds all state.
        self.key_model_metrics_df: pd.DataFrame | None = None
        self.lock = asyncio.Lock()

        # --- State Management ---
        self.is_ready = False
        self._background_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

        # --- Timestamps for Resets ---
        self.last_minute_reset_ts: float = 0.0
        self.last_day_reset_date: date | None = None
        self.last_db_commit_ts: float = 0.0
        
        # --- Key Caching (for get_key stickiness) ---
        self.last_key_cache: dict[str, str] = {}
        self.last_key_cache_ts: dict[str, float] = {}

    @classmethod
    async def create(
        cls,
        async_session_maker: async_sessionmaker,
        rate_limit_data: dict,
        commit_interval: int,
        minute_reset_interval: int,
        tz_name: str,
    ):
        """
        Async factory for creating and initializing the KeyManager.
        This runs the initial data load and starts background tasks.
        """
        log.info("Creating and initializing KeyManager...")
        manager = cls(
            async_session_maker,
            rate_limit_data,
            commit_interval,
            minute_reset_interval,
            tz_name,
        )

        # 1. Load data from the database
        await manager._load_data()

        # 2. Run initial reset to set timestamps
        await manager.reset_usage(level="all")

        # 3. Start background resetter
        manager._background_task = asyncio.create_task(manager.reset_usage_bg())

        # 4. Set ready state
        manager.is_ready = True
        log.info("KeyManager is ready.")
        return manager

    async def shutdown(self):
        """
        Gracefully shuts down the KeyManager.
        """
        log.info("Shutting down KeyManager...")
        # 1. Signal background task to stop
        self._stop_event.set()

        # 2. Wait for task to finish
        if self._background_task:
            await self._background_task

        # 3. Perform one final commit to the DB
        log.info("Performing final database commit...")
        await self._commit_to_db()
        log.info("KeyManager shutdown complete.")

    async def _load_data(self):
        """
        Loads all key/model configurations from the DB and merges
        them with the provided rate_limit_data.
        
        This populates self.key_model_metrics_df
        """
        log.info("Loading key and usage data from database...")
        
        # --- This is a complex operation based on our previous conversation ---
        # 1. Flatten the rate_limit_data
        records = []
        for api_key, models in self.rate_limit_data.items():
            for model_name, limits in models.items():
                records.append({
                    "api_key": api_key,
                    "model_name": model_name,
                    **limits,
                })
        if not records:
            raise ValueError("rate_limit_data is empty!")
            
        df = pd.DataFrame(records)
        df.set_index(["api_key", "model_name"], inplace=True)

        # 2. Load persisted data from DB (is_active, today_requests)
        # We assume a table 'model_key_usage'
        query = """
        SELECT api_key, model_name, is_active, today_requests 
        FROM model_key_usage
        """
        
        try:
            async with self.db_maker() as session:
                result = await session.execute(text(query))
                db_data = result.fetchall()
            
            if db_data:
                db_df = pd.DataFrame(db_data, columns=["api_key", "model_name", "is_active", "today_requests"])
                db_df.set_index(["api_key", "model_name"], inplace=True)
                
                # 3. Join DB data onto our rate limit data
                df = df.join(db_df, how="left")
                
                # Fill defaults for keys that are in rate_limit_data but not DB
                df["is_active"] = df["is_active"].fillna(True).astype(bool)
                df["today_requests"] = df["today_requests"].fillna(0).astype(int)
            else:
                # No data in DB, just use defaults
                log.warning("No data found in 'model_key_usage'. Using defaults.")
                df["is_active"] = True
                df["today_requests"] = 0

        except Exception as e:
            log.error(f"Failed to load from database: {e}. Using defaults.")
            df["is_active"] = True
            df["today_requests"] = 0

        # 4. Add all transient/tracking columns
        df.rename(columns={"today_requests": "rpd"}, inplace=True)
        df["rpm"] = 0
        df["tpm"] = 0
        df["is_exhausted"] = False
        df["last_used_timestamp"] = 0.0
        
        # 5. Lock and set the main DataFrame
        async with self.lock:
            self.key_model_metrics_df = df
            
        log.info(f"Successfully loaded {len(df)} key/model combinations.")

    async def get_key(self, model_name: str) -> str:
        """
        Finds and returns the best available API key for a given model.
        Implements "sticky" key caching within a 1-minute window.
        """
        if not self.is_ready:
            raise Exception("KeyManager is not ready.")

        now = datetime.now(self.tz).timestamp()
        
        async with self.lock:
            if self.key_model_metrics_df is None:
                raise Exception("DataFrame not initialized.")

            # 1. --- Fast Path (Cache Check) ---
            last_key = self.last_key_cache.get(model_name)
            last_ts = self.last_key_cache_ts.get(model_name, 0.0)
            
            # Check if cache is valid for this minute
            if last_key and last_ts >= self.last_minute_reset_ts:
                try:
                    idx = (last_key, model_name)
                    # Explicitly type hint metrics_row as pd.Series
                    metrics_row: pd.Series = self.key_model_metrics_df.loc[idx] # type: ignore
                    
                    # Check if key is still usable
                    rpm_left = int(metrics_row["max_rpm"]) - int(metrics_row["rpm"]) # type: ignore
                    if bool(metrics_row["is_active"]) and not bool(metrics_row["is_exhausted"]) and rpm_left >= 1: # type: ignore
                        # Cache Hit!
                        return last_key
                except KeyError:
                    # Key might have been removed, fall through to full path
                    pass 

            # 2. --- Full Path (Query) ---
            df = self.key_model_metrics_df.copy()
            
            # Calculate dynamic values
            df["tpm_left"] = df["max_tpm"] - df["tpm"]
            df["rpm_left"] = df["max_rpm"] - df["rpm"]
            df["rpd_usage_percent"] = (df["rpd"] / df["max_rpd"]).fillna(0)

            # Filter for the specific model
            try:
                candidates = df.xs(model_name, level="model_name").copy()
            except KeyError:
                raise Exception(f"No keys configured for model: {model_name}")

            # Apply filters
            candidates = candidates.query(
                "is_active == True and "
                "is_exhausted == False and "
                "rpm_left >= 1 and "
                "rpd_usage_percent < 0.3"
            )

            if candidates.empty:
                raise Exception(f"No available keys for model {model_name}.")

            # Sort by highest TPM left
            candidates_sorted = candidates.sort_values(by="tpm_left", ascending=False)
            
            # Get the API key (which is the index)
            best_key_string = candidates_sorted.index[0]

            # 3. --- Update Cache ---
            self.last_key_cache[model_name] = best_key_string
            self.last_key_cache_ts[model_name] = now

            return best_key_string

    async def update_usage(
        self,
        key_value: str,
        model_name: str,
        tokens_used: int,
        error: bool = False,
        error_type: str | None = None,
    ):
        """
        Updates the in-memory DataFrame with new usage data.
        This is a fast, async, in-memory-only operation.
        """
        async with self.lock:
            if self.key_model_metrics_df is None:
                return # Not ready
                
            try:
                idx = (key_value, model_name)
                
                if error:
                    if error_type == "permanent":
                        # Deactivate key for ALL models
                        self.key_model_metrics_df.loc[key_value, "is_active"] = False
                    elif error_type == "temporary":
                        # Exhaust this specific model
                        self.key_model_metrics_df.loc[idx, "is_exhausted"] = True
                    return

                # --- Update metrics on success ---
                self.key_model_metrics_df.loc[idx, "rpd"] += 1 # type: ignore
                self.key_model_metrics_df.loc[idx, "rpm"] += 1 # type: ignore
                self.key_model_metrics_df.loc[idx, "tpm"] += tokens_used # type: ignore
                self.key_model_metrics_df.loc[idx, "last_used_timestamp"] = datetime.now(self.tz).timestamp() # type: ignore
                
                # Check for self-exhaustion
                if self.key_model_metrics_df.loc[idx, "rpm"] >= self.key_model_metrics_df.loc[idx, "max_rpm"]: # type: ignore
                    self.key_model_metrics_df.loc[idx, "is_exhausted"] = True # type: ignore

            except KeyError:
                log.warning(f"Attempted to update non-existent key/model: {key_value}/{model_name}")
            except Exception as e:
                log.error(f"Error in update_usage: {e}")

    async def reset_usage(self, level: str = "all"):
        """
        The core LOGIC for resetting metrics in the DataFrame.
        This method is called by the background task.
        """
        async with self.lock:
            if self.key_model_metrics_df is None:
                return # Not ready

            now = datetime.now(self.tz)
            
            if level == "all" or level == "minute":
                log.info("Resetting minute-level metrics (rpm, tpm, exhausted)...")
                self.key_model_metrics_df["rpm"] = 0
                self.key_model_metrics_df["tpm"] = 0
                self.key_model_metrics_df["is_exhausted"] = False
                
                # Clear the 1-minute key cache
                self.last_key_cache.clear()
                self.last_key_cache_ts.clear()
                
                self.last_minute_reset_ts = now.timestamp()
            
            if level == "all" or level == "day":
                log.info("Resetting day-level metrics (rpd)...")
                self.key_model_metrics_df["rpd"] = 0
                self.last_day_reset_date = now.date()

    async def _commit_to_db(self):
        """
        Saves the current state (rpd, is_active) to the database.
        """
        if self.key_model_metrics_df is None:
            return

        # 1. Get a thread-safe copy of the data to save
        async with self.lock:
            df_to_save = self.key_model_metrics_df[["rpd", "is_active"]].copy()
        
        # 2. Iterate and save
        log.info(f"Committing {len(df_to_save)} records to database...")
        try:
            async with self.db_maker() as session:
                for (api_key, model_name), row in df_to_save.iterrows(): # type: ignore
                    # Use an "upsert" (INSERT OR REPLACE)
                    stmt = text("""
                        INSERT INTO model_key_usage (api_key, model_name, today_requests, is_active)
                        VALUES (:api_key, :model_name, :rpd, :is_active)
                        ON CONFLICT(api_key, model_name) DO UPDATE SET
                            today_requests = :rpd,
                            is_active = :is_active
                    """)
                    await session.execute(stmt, {
                        "api_key": api_key,
                        "model_name": model_name,
                        "rpd": int(row["rpd"]),
                        "is_active": bool(row["is_active"])
                    })
                await session.commit()
            self.last_db_commit_ts = datetime.now(self.tz).timestamp()
            log.info("Database commit successful.")
        except Exception as e:
            log.error(f"Database commit failed: {e}")

    async def reset_usage_bg(self):
        """
        The main background task that runs the reset and commit loops.
        """
        log.info("Background reset/commit task started.")
        while not self._stop_event.is_set():
            try:
                now = datetime.now(self.tz)
                now_ts = now.timestamp()
                
                # 1. Check for Minute Reset
                if (now_ts - self.last_minute_reset_ts) >= self.minute_reset_interval_sec:
                    await self.reset_usage(level="minute")
                
                # 2. Check for Day Reset
                if self.last_day_reset_date is None or now.date() > self.last_day_reset_date:
                    await self.reset_usage(level="day")
                    
                # 3. Check for DB Commit
                if (now_ts - self.last_db_commit_ts) >= self.commit_interval_sec:
                    await self._commit_to_db()

                # Wait for 1 second or until stop_event is set
                await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
            
            except asyncio.TimeoutError:
                continue # This is the normal loop
            except Exception as e:
                log.error(f"Error in background task: {e}")
                # Don't crash the loop, just wait and retry
                await asyncio.sleep(5) 
                
        log.info("Background reset/commit task stopped.")