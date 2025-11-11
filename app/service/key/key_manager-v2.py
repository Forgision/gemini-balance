import asyncio
import itertools
import time
from typing import AsyncGenerator, Optional
import pandas as pd
import pytz
from datetime import datetime, date, timedelta
from sqlalchemy import select, MetaData, Column, Integer, String, DateTime, Boolean, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.config.config import settings
from app.log.logger import get_key_manager_logger
from app.database.models import UsageStats
from app.service.key import rate_limits
from app.service.key.rate_limits import scrape_gemini_rate_limits


DATABASE_URL = settings.KEY_MATRIX_DB_URL
GEMINI_RATE_LIMIT_URL = "https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits"

# Configure logging
logger = get_key_manager_logger()


engine = create_async_engine(
            DATABASE_URL,
            echo = False,
            pool_size = 50,
            max_overflow = 100,
            pool_timeout = 10
        )

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession,
        expire_on_commit=False, autoflush=False)

# Create metadata object
metadata = MetaData()

# Create base class
class Base(DeclarativeBase):
    metadata = metadata
    

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields a DB session; use in FastAPI dependencies."""
    logger.debug("Acquiring database session.")
    try:
        async with AsyncSessionLocal() as session:
            yield session
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        raise
    finally:
        logger.debug("Database session released.")
    
    
async def init_database():
    logger.debug("Initializing UsageMatixdb")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.debug("Initializing UsageMatixdb done")


class UsageMatrix(Base):
    """
    Usage statistics table
    """

    __tablename__ = "t_usage_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    api_key = Column(String(100), nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    rpm = Column(Integer, nullable=False, default=0)
    tpm = Column(Integer, nullable=False, default=0)
    rpd = Column(Integer, nullable=False, default=0)
    minute_reset_time = Column(DateTime, nullable=True)
    day_reset_time = Column(DateTime, nullable=True)
    vertex_key = Column(Boolean, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    is_exhausted = Column(Boolean, nullable=False, default=False)
    last_used = Column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    def __repr__(self):
        return (
            f"<UsageStats(api_key='{self.api_key}', "
            f"model_name='{self.model_name}', "
            f"rpm='{self.rpm}', "
            f"tpm='{self.tpm}', "
            f"rpd='{self.rpd}', "
            f"minute_reset_time='{self.minute_reset_time}', "
            f"day_reset_time='{self.day_reset_time}', "
            f"vertex_key='{self.vertex_key}', "
            f"is_active='{self.is_active}', "
            f"is_exhausted='{self.is_exhausted}', "
            f"last_used='{self.last_used}')>"
        )


def _get_real_model(model: str) -> str:
    if "-search" in model and "-non-thinking" in model:
        model = model[:-20]
    if model.endswith("-search"):
        model = model[:-7]
    if model.endswith("-image"):
        model = model[:-6]
    if model.endswith("-non-thinking"):
        model = model[:-13]
    return model


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
        ['rpm', 'tpm', 'rpd', 'max_rpm', 'max_tpm', 'max_rpd', "is_active", "is_exhausted", "last_used"]
    ]

    def __init__(
        self,
        api_keys: list[str], vertex_api_keys: list[str],
        async_session_maker: async_sessionmaker,
        rate_limit_data: Optional[dict] = None,
        minute_reset_interval: Optional[int] = None
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
        self.rate_limit_data = rate_limit_data
        self.rate_limit_models: list[str] = list(rate_limit_data.keys()) if rate_limit_data else []
        self.tz = pytz.timezone(zone="UTC")
        self.now = lambda: datetime.now(self.tz)
        self.now_minute = lambda: self.now().replace(second=0, microsecond=0)
        self.now_day = lambda: self.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.minute_reset_interval_sec = minute_reset_interval or 60 # default 60 seconds
        # self.scheduler = scheduler = AsyncIOScheduler(self.tz)    #TODO: use this schedule to sechdule reset

        # --- Core Data ---
        # This DataFrame is the "brain" and holds all state.
        self.df: pd.DataFrame = pd.DataFrame()
        self.lock = asyncio.Lock()

        # --- State Management ---
        self.is_ready = False
        self._background_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

        # --- Timestamps for Resets ---
        self.last_minute_reset_ts: datetime = self.now_minute()
        self.last_day_reset_ts: datetime = self.now_day()
        self.last_db_commit_ts: datetime = self.now()
        
    def _model_normalization(self, model_name: str) -> tuple[bool, str]:
        """
        Checks if an input string starts with any prefix from a given list.
        If a match is found, it returns the longest matching prefix.
        If no match is found, it returns the original string.
        """
        if not isinstance(model_name, str) or not model_name:
            raise TypeError("Input must be a non-empty string.")

        if self.rate_limit_models is None or not isinstance(self.rate_limit_models, list) or len(self.rate_limit_models) < 1:
            logger.warning("Rate Limits data is not found")
            return (False, model_name)
        
        for prefix in sorted(self.rate_limit_models, key=len, reverse=True):
            if model_name in prefix:
                # As soon as we find a match (which will be the longest one), return it.
                return (True, prefix)
                
        # If the loop finishes without finding any match, return the original string.
        return (False, model_name)
        
    async def _check_ready(self):
        error_msg = ""
        if self.rate_limit_data is None or not isinstance(self.rate_limit_data, dict) or len(self.rate_limit_data) < 1:
            error_msg += "Rate Limits data is not found"
        
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            error_msg += "\nDataFrame not initialized."
        else:
            missing_columns = [ column for column in ['rpm', 'tpm', 'rpd', 'max_rpm', 'max_tpm', 'max_rpd', 'rpm_left', 'tpm_left', 'rpd_left'] if column not in self.df.columns]
            if missing_columns:
                error_msg += f"\nRequired columns {missing_columns} not found"
        
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
        
        # Checking rate limit data
        if not self.rate_limit_data:
            raise ValueError("Rate limits data not found")
        
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
                        "max_rpm": limits['RPM'],
                        "tpm": 0,
                        "max_tpm": limits['TPM'],
                        "rpd": 0,
                        "max_rpd": limits['RPD'],
                        "minute_reset_time": self.now_minute(),
                        "day_reset_time": self.now_day(),
                        "last_used": self.now() - timedelta(2),  # set previous second day as last used
                        "is_vertex_key": False,
                        "is_active": True,
                        "is_exhausted": False
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
                            "max_rpm": limits['RPM'],
                            "tpm": 0,
                            "max_tpm": limits['TPM'],
                            "rpd": 0,
                            "max_rpd": limits['RPD'],
                            "minute_reset_time": self.now_minute(),
                            "day_reset_time": self.now_day(),
                            "last_used": self.now() - timedelta(2),  # set previous second day as last used
                            "is_vertex_key": True,
                            "is_active": True,
                            "is_exhausted": False                            
                        }
                    )
                    
        # Checking that all api keys have incorate in data
        if len(defaults) < len(self.api_keys) + len(self.vertex_api_keys):
            raise ValueError("Defaults is not created for each keys")

        # Updating dataframe
        async with self.lock:
            self.df = pd.DataFrame(defaults)
            self.df.set_index(self._INDEX_LEVEL, inplace=True, drop=True)
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
                        "minute_reset_time": item.minute_reset_time,
                        "day_reset_time": item.day_reset_time,
                        "last_used": item.last_used,
                        "is_vertex_key": item.vertex_key,
                        "is_active": item.is_active,
                        "is_exhausted": item.is_exhausted
                    } for item in db_data
                ]

                df = pd.DataFrame(db_data_dict)
                df.set_index(self._INDEX_LEVEL, inplace=True, drop=True)
                    
                async with self.lock:
                    # self.df is guaranteed to be initialized by _load_default before this method is called.
                    # Ensure self.df is also indexed for proper update.
                    if isinstance(self.df, pd.DataFrame):
                        # Update existing entries in self.df with values from db_df.
                        # This will only update rows where the index (api_key, model_name) matches.
                        # Columns present in self.df but not in db_df (e.g., 'vertex_key') will remain unchanged.
                        self.df.update(df)
                        logger.info(f"Successfully merged {len(df)} records from database into self.df.")
                    else:
                        self.df = df
                        
                    # setting reset times
                    max_minute_reset_dt = self.df['minute_reset_time'].max()
                    if pd.isna(max_minute_reset_dt):
                        self.last_minute_reset_ts = datetime.now(self.tz)
                    else:
                        self.last_minute_reset_ts = max_minute_reset_dt
                    
                    max_day_reset_dt = self.df['day_reset_time'].max()
                    if not pd.isna(max_day_reset_dt):
                        self.last_day_reset_ts = max_day_reset_dt
                    
            else:
                # No data in DB, self.df remains as initialized by _load_default.
                logger.warning("No data found in 't_usage_stats'. Using defaults from _load_default.")

        except Exception as e:
            logger.error(f"Failed to load from database: {e}. Keeping defaults from _load_default.")

    async def _set_available_usage(self):
        if not self.is_ready:
            await self._check_ready()
            
        missing_columns = [ column for column in ['rpm', 'tpm', 'rpd', 'max_rpm', 'max_tpm', 'max_rpd'] if column not in self.df.columns]
        if missing_columns:
            raise TypeError(f"Required columns {missing_columns} not found")
        
        self.df['rpm_left'] = self.df['max_rpm'] -  self.df['rpm']
        self.df['tpm_left'] = self.df['max_tpm'] -  self.df['tpm']
        self.df['rpd_left'] = self.df['max_rpd'] -  self.df['rpd']
        
        # Ensure no negative values for 'left' columns
        self.df['rpm_left'] = self.df['rpm_left'].clip(lower=0)
        self.df['tpm_left'] = self.df['tpm_left'].clip(lower=0)
        self.df['rpd_left'] = self.df['rpd_left'].clip(lower=0)
        
    async def _set_exhausted_flags(self):
        if not self.is_ready:
            await self._check_ready()
            
        missing_columns = [ column for column in ['rpm', 'tpm', 'rpd', 'max_rpm', 'max_tpm', 'max_rpd'] if column not in self.df.columns]
        if missing_columns:
            raise TypeError(f"Required columns {missing_columns} not found")
        
        self.df['is_exhausted'] = False # Reset all to False first
        
        self.df['is_exhausted'] = (
            (self.df['rpm'] >= self.df['max_rpm']) |
            (self.df['tpm'] >= self.df['max_tpm']) |
            (self.df['rpd'] >= self.df['max_rpd'])
        )

    async def _ensure_numeric_columns(self):
        """Ensures that the columns used for comparison are numeric."""
        numeric_cols = ['rpm', 'tpm', 'rpd', 'max_rpm', 'max_tpm', 'max_rpd']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)

    async def _on_update_usage(self):
        """
        Callback for when usage is updated.
        Currently not used.
        """
        await self._set_available_usage()
        await self._set_exhausted_flags()
        await self._commit_to_db()
        
    async def init(self,
        async_session_maker: async_sessionmaker,
        rate_limit_data: dict,
        minute_reset_interval: int
    ) -> bool:
        """
        Async factory for creating and initializing the KeyManager.
        This runs the initial data load and starts background tasks.
        """
        try:
            # create database
            await init_database()
            
            self.db_maker = async_session_maker or self.db_maker
            self.rate_limit_data = rate_limit_data or self.rate_limit_data
            self.minute_reset_interval_sec = minute_reset_interval or self.minute_reset_interval_sec
            
            logger.info("Initializing KeyManager...")
            
            # get rate limit data
            rate_limit_data = scrape_gemini_rate_limits(GEMINI_RATE_LIMIT_URL)
            if not rate_limit_data:
                raise ValueError("rate_limit_data is empty!")
            
            self.rate_limit_data = rate_limit_data['Free Tier'].copy()
            self.rate_limit_models = sorted(list(self.rate_limit_data.keys()), key=len, reverse=True)
            del rate_limit_data
            
            # load defaults
            await self._load_default()
            
            # 1. Load data from the database
            await self._load_from_db()

            await self._ensure_numeric_columns()

            # 2. Run initial reset to set timestamps
            await self.reset_usage()

            # 3. Start background resetter
            self._background_task = asyncio.create_task(self.reset_usage_bg())

            # 4. Set ready state
            if await self._check_ready():
                self.is_ready = True
            logger.info("KeyManager is ready.")
            return True
        except Exception as e:
            logger.exception("KeyManager initialization is failed")
            return False

    async def shutdown(self):
        """
        Gracefully shuts down the KeyManager.
        """
        logger.info("Shutting down KeyManager...")
        # 1. Signal background task to stop
        self._stop_event.set()

        # 2. Wait for task to finish
        if self._background_task:
            await self._background_task

        # 3. Perform one final commit to the DB
        logger.info("Performing final database commit...")
        await self._commit_to_db()
        logger.info("KeyManager shutdown complete.")

    async def get_key(self, model_name: str) -> str:
        """
        Finds and returns the best available API key for a given model.
        Implements "sticky" key caching within a 1-minute window.
        """
        model_name = _get_real_model(model_name)
        
        if not self.is_ready:
            await self._check_ready()
        
        async with self.lock:
            # Filter for the specific model
            try:
                candidates = self.df.xs(model_name, level="model_name").copy()
            except KeyError:
                raise Exception(f"No keys configured for model: {model_name}")

            # Apply filters
            candidates = candidates.query(
                "is_active == True and "
                "is_exhausted == False and "
                "rpm_left >= 1 and "
                "rpd_left >= 1"
            )
            
            # Check data is not empty
            if candidates.empty:
                raise Exception(f"No available keys for model {model_name}.")

            # Sort by tpm_left descending
            candidates.sort_values(by=["tpm_left"], ascending=False, inplace=True)
            
            # Get the API key (which is the 3rd level of the index after filtering by model)
            best_key_string = candidates.index[0][2]

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
        Updates the in-memory DataFrame with new usage data.
        This is a fast, async, in-memory-only operation.
        """
        async with self.lock:
            if not self.is_ready:
                await self._check_ready()
                
            try:
                idx = (model_name, is_vertex_key, key_value)
                
                if error:
                    if error_type == "permanent":
                        # Deactivate key for ALL models
                        self.df.loc[self.df.index.get_level_values('api_key') == key_value, 'is_active'] = False
                    elif error_type == "429":
                        # Exhaust this specific model
                        self.df.loc[idx, "is_exhausted"] = True
                    return

                # --- Update metrics on success ---
                rpd_val = pd.to_numeric(self.df.loc[idx, "rpd"], errors='coerce')
                rpm_val = pd.to_numeric(self.df.loc[idx, "rpm"], errors='coerce')
                tpm_val = pd.to_numeric(self.df.loc[idx, "tpm"], errors='coerce')
                
                self.df.loc[idx, "rpd"] = int(rpd_val) + 1
                self.df.loc[idx, "rpm"] = int(rpm_val) + 1
                self.df.loc[idx, "tpm"] = int(tpm_val) + tokens_used
                self.df.loc[idx, "last_used"] = self.now()

                # Calculate updated available usage
                await self._on_update_usage()
                return True
            except KeyError:
                logger.warning(f"Attempted to update non-existent key/model: {key_value}/{model_name}")
                return False
            except Exception as e:
                logger.error(f"Error in update_usage: {e}")
                return False

    async def reset_usage(self):
        """
        The core LOGIC for resetting metrics in the DataFrame.
        This method is called by the background task.
        """
        try:
            async with self.lock:
                if not self.is_ready:
                    await self._check_ready()

                now_minute = self.now_minute()
                now_day = self.now_day()
                
                if self.last_minute_reset_ts + timedelta(seconds=59) < now_minute:
                    logger.debug("Resetting minute-level metrics (rpm, tpm, exhausted)...")
                    self.df["rpm"] = 0
                    self.df["tpm"] = 0
                    self.df["is_exhausted"] = False
                    self.last_minute_reset_ts = now_minute

                if self.last_day_reset_ts + timedelta(days=1) < now_day:
                    logger.debug("Resetting day-level metrics (rpd)...")
                    self.df["rpd"] = 0
                    self.last_day_reset_ts = now_day
                
                await self._on_update_usage()
                return True
        except Exception as e:
            logger.exception(f"Error in reset_usage: {e}")
            return False
    
    async def _commit_to_db(self):
        """
        Saves the current state (rpd, is_active) to the database.
        """
        if self.df is None:
            return

        # 1. Get a thread-safe copy of the data to save
        async with self.lock:
            df_to_save = self.df.copy()
        
        # 2. Iterate and save
        logger.info(f"Committing {len(df_to_save)} records to database...")
        try:
            async with self.db_maker() as session:
                for (model_name, is_vertex_key, api_key), row in df_to_save.iterrows(): # type: ignore
                    # Use a more general "upsert" method
                    stmt = select(UsageMatrix).where(
                        UsageMatrix.api_key == api_key,
                        UsageMatrix.model_name == model_name
                    )
                    result = await session.execute(stmt)
                    instance = result.scalars().first()

                    if instance:    # Update existing record
                        instance.rpm = int(row["rpm"]) # type: ignore
                        instance.tpm = int(row["tpm"]) # type: ignore
                        instance.rpd = int(row["rpd"]) # type: ignore
                        instance.minute_reset_time = self.last_minute_reset_ts # type: ignore
                        instance.day_reset_time = self.last_day_reset_ts # type: ignore
                        instance.is_exhausted = bool(row["is_exhausted"]) # type: ignore
                        instance.last_used = row["last_used"] # type: ignore
                        instance.is_active = bool(row["is_active"]) # type: ignore
                    else:   # Insert new record
                        instance = UsageMatrix(
                            api_key=api_key,
                            model_name=model_name,
                            rpm=int(row["rpm"]),
                            tpm=int(row["tpm"]),
                            rpd=int(row["rpd"]),
                            minute_reset_time=self.last_minute_reset_ts,
                            day_reset_time=self.last_day_reset_ts,
                            vertex_key=is_vertex_key,
                            is_active=bool(row["is_active"]),
                            is_exhausted=bool(row["is_exhausted"]),
                            last_used=row["last_used"]
                        )
                        session.add(instance)
                await session.commit()
                
            self.last_db_commit_ts = self.now()
            logger.debug("Database commit successful.")
        except Exception as e:
            logger.error(f"Database commit failed: {e}")

    async def reset_usage_bg(self):
        """
        The main background task that runs the reset and commit loops.
        """
        logger.info("Background reset/commit task started.")
        while not self._stop_event.is_set():
            try:
                if not self.is_ready:
                    await asyncio.sleep(1)
                    await self._check_ready()
                
                reset: bool = await self.reset_usage()
                
                # 3. Check for DB Commit
                if reset:
                    await self._on_update_usage()

                # Wait for 1 second or until stop_event is set
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass # This is the normal loop
            
            except Exception as e:
                logger.exception(f"Error in background task: {e}")
                # Don't crash the loop, just wait and retry
                await asyncio.sleep(5) 
                
        logger.info("Background reset/commit task stopped.")