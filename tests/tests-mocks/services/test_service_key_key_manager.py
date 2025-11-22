"""
Comprehensive pytest tests for KeyManager v2 class.

This test suite covers:
- Initialization and setup
- API key retrieval logic
- Usage tracking and updates
- Rate limit enforcement
- Reset mechanisms
- Database persistence
- Model normalization
- Background task management
- Error handling
"""

import pytest
import pytest_asyncio
import pandas as pd
import asyncio
from datetime import timedelta
from unittest.mock import patch
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.service.key.key_manager import KeyManager
from app.database.models import UsageMatrix
from app.database.connection import Base
from app.database.connection import get_db


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def test_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        pool_pre_ping=False,  # Disable pre-ping for faster tests
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Properly dispose of the engine and close all connections
    await engine.dispose(close=True)
    # Give aiosqlite threads time to finish cleanup
    await asyncio.sleep(0.1)


@pytest_asyncio.fixture
async def test_session_maker(test_engine):
    """Create a session maker for testing."""
    return async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest.fixture
def mock_rate_limit_data():
    """Mock rate limit data for testing."""
    return {
        "gemini-1.5-flash": {"RPM": 5, "TPM": 1000000, "RPD": 20},
        "gemini-1.5-pro": {"RPM": 3, "TPM": 500000, "RPD": 10},
        "gemini-2.0-flash": {"RPM": 2, "TPM": 2000000, "RPD": 5},
    }


@pytest.fixture
def sample_api_keys():
    """Sample API keys for testing."""
    return ["test_key_1", "test_key_2", "test_key_3"]


@pytest.fixture
def sample_vertex_keys():
    """Sample Vertex API keys for testing."""
    return ["vertex_key_1", "vertex_key_2"]


@pytest_asyncio.fixture
async def key_manager(
    sample_api_keys,
    sample_vertex_keys,
    test_session_maker,
    mock_rate_limit_data,
):
    """Create a KeyManager instance for testing."""
    # Create a patcher that will be active for the entire fixture lifecycle
    patcher = patch(
        "app.service.key.key_manager_v2.scrape_gemini_rate_limits",
        return_value={"Free Tier": mock_rate_limit_data},
    )
    patcher.start()

    try:
        km = KeyManager(
            api_keys=sample_api_keys,
            vertex_api_keys=sample_vertex_keys,
            async_session_maker=test_session_maker,
            rate_limit_data=mock_rate_limit_data,
            minute_reset_interval=60,  # 60 seconds - long enough to not trigger during tests
        )

        await km.init()

        # Wait a tiny bit to ensure initialization is complete
        await asyncio.sleep(0.01)

        yield km

        # Cleanup - shutdown should handle background task cleanup
        try:
            # Signal shutdown first
            km._stop_event.set()

            # Wait for background task to finish (with timeout)
            if km._background_task and not km._background_task.done():
                try:
                    await asyncio.wait_for(km._background_task, timeout=2.0)
                except asyncio.TimeoutError:
                    # Force cancel if it doesn't stop in time
                    km._background_task.cancel()
                    try:
                        await km._background_task
                    except asyncio.CancelledError:
                        pass

            await km.shutdown()
        except Exception as e:
            # Log but don't fail if shutdown has issues
            import logging

            logging.warning(f"Error during KeyManager shutdown: {e}")

        # Give aiosqlite threads minimal time to finish
        await asyncio.sleep(0.05)
    finally:
        patcher.stop()


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.asyncio
async def test_init_database(test_engine):
    """Test database initialization."""
    async with test_engine.begin() as conn:
        # Verify tables are created
        result = await conn.run_sync(
            lambda sync_conn: sync_conn.dialect.has_table(sync_conn, "t_usage_matrix")
        )
        assert result is True


@pytest.mark.asyncio
async def test_get_db():
    """Test the get_db dependency."""
    session_count = 0
    async for session in get_db():
        assert isinstance(session, AsyncSession)
        session_count += 1
        break  # Only test first session
    assert session_count == 1


@pytest.mark.asyncio
async def test_key_manager_initialization(
    sample_api_keys,
    sample_vertex_keys,
    test_session_maker,
    mock_rate_limit_data,
):
    """Test KeyManager initialization."""
    with patch(
        "app.service.key.key_manager_v2.scrape_gemini_rate_limits",
        return_value={"Free Tier": mock_rate_limit_data},
    ):
        km = KeyManager(
            api_keys=sample_api_keys,
            vertex_api_keys=sample_vertex_keys,
            async_session_maker=test_session_maker,
            rate_limit_data=mock_rate_limit_data,
        )

        result = await km.init()

        assert result is True
        assert km.is_ready is True
        assert isinstance(km.df, pd.DataFrame)
        assert not km.df.empty
        assert km.api_keys == sample_api_keys
        assert km.vertex_api_keys == sample_vertex_keys

        await km.shutdown()


@pytest.mark.asyncio
async def test_key_manager_initialization_no_api_keys(
    test_session_maker,
    mock_rate_limit_data,
):
    """Test KeyManager initialization fails with no API keys."""
    with patch(
        "app.service.key.key_manager_v2.scrape_gemini_rate_limits",
        return_value={"Free Tier": mock_rate_limit_data},
    ):
        km = KeyManager(
            api_keys=[],
            vertex_api_keys=[],
            async_session_maker=test_session_maker,
            rate_limit_data=mock_rate_limit_data,
        )

        # Expecting RuntimeError to be raised because if no api keys are provided, the KeyManager.load_default will raise ValueError, on base of that KeyManager.init will raise RuntimeError.
        with pytest.raises(RuntimeError):
            result = await km.init()

            assert result is False
            assert km.is_ready is False


@pytest.mark.asyncio
async def test_key_manager_dataframe_structure(key_manager):
    """Test that the DataFrame has the expected structure."""
    df = key_manager.df

    # Check index levels
    assert df.index.names == ["model_name", "is_vertex_key", "api_key"]

    # Check columns
    expected_columns = [
        "rpm",
        "tpm",
        "rpd",
        "max_rpm",
        "max_tpm",
        "max_rpd",
        "minute_reset_time",
        "day_reset_time",
        "is_active",
        "is_exhausted",
        "last_used",
        "rpm_left",
        "tpm_left",
        "rpd_left",
    ]
    for col in expected_columns:
        assert col in df.columns


@pytest.mark.asyncio
async def test_load_default_creates_entries_for_all_keys(key_manager):
    """Test that _load_default creates entries for all keys and models."""
    df = key_manager.df

    # Should have entries for all combinations of keys and models
    num_models = len(key_manager.rate_limit_data)
    num_keys = len(key_manager.api_keys) + len(key_manager.vertex_api_keys)
    expected_rows = num_models * num_keys

    assert len(df) == expected_rows


# ============================================================================
# Model Normalization Tests
# ============================================================================


@pytest.mark.asyncio
async def test_model_normalization_exact_match(key_manager):
    """Test model normalization with exact match."""
    normalized = key_manager._model_normalization("gemini-1.5-flash")
    assert normalized == "gemini-1.5-flash"


@pytest.mark.asyncio
async def test_model_normalization_prefix_match(key_manager):
    """Test model normalization with prefix match."""
    normalized = key_manager._model_normalization("gemini-1.5-flash-001")
    assert normalized == "gemini-1.5-flash"


@pytest.mark.asyncio
async def test_model_normalization_no_match(key_manager):
    """Test model normalization with no match."""
    normalized = key_manager._model_normalization("unknown-model")
    assert normalized == "unknown-model"


@pytest.mark.asyncio
async def test_model_normalization_longest_prefix(key_manager):
    """Test that model normalization returns the longest matching prefix."""
    # If we have both "gemini-1.5" and "gemini-1.5-flash" in rate limits,
    # it should match "gemini-1.5-flash" for "gemini-1.5-flash-001"
    normalized = key_manager._model_normalization("gemini-1.5-flash-search")
    assert normalized == "gemini-1.5-flash"


@pytest.mark.asyncio
async def test_model_normalization_invalid_input(key_manager):
    """Test model normalization with invalid input."""
    with pytest.raises(TypeError):
        key_manager._model_normalization("")

    with pytest.raises(TypeError):
        key_manager._model_normalization(None)


# ============================================================================
# Get Key Tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_key_returns_valid_key(
    key_manager,
):
    """Test that get_key returns a valid API key."""
    # Ensure the KeyManager is ready before calling get_key
    assert key_manager.is_ready is True

    await key_manager.update_usage(
        model_name="gemini-1.5-flash",
        key_value="test_key_1",
        is_vertex_key=False,
        tokens_used=1000,
    )
    await key_manager.update_usage(
        model_name="gemini-1.5-flash",
        key_value="test_key_2",
        is_vertex_key=False,
        tokens_used=1000,
    )

    key = await asyncio.wait_for(
        key_manager.get_key("gemini-1.5-flash", is_vertex_key=False), timeout=5.0
    )
    assert key in key_manager.api_keys
    assert (
        key == "test_key_3"
    )  # test_key_3 is the first key in the list as test_key_1 and test_key_2 are already used


@pytest.mark.asyncio
async def test_get_key_returns_vertex_key(key_manager):
    """Test that get_key can return a vertex key."""
    key = await key_manager.get_key("gemini-1.5-flash", is_vertex_key=True)
    assert key in key_manager.vertex_api_keys


@pytest.mark.asyncio
async def test_get_key_prefers_most_available(key_manager):
    """Test that get_key prefers keys with most TPM left."""
    model = "gemini-1.5-flash"

    # Use key_1 heavily
    await key_manager.update_usage(
        model_name=model,
        key_value="test_key_1",
        is_vertex_key=False,
        tokens_used=500000,
    )

    # Use key_2 lightly
    await key_manager.update_usage(
        model_name=model,
        key_value="test_key_2",
        is_vertex_key=False,
        tokens_used=1000,
    )

    # Should prefer key_3 (unused) or key_2 (lightly used)
    key = await key_manager.get_key(model, is_vertex_key=False)
    assert key in ["test_key_2", "test_key_3"]


@pytest.mark.asyncio
async def test_get_key_skips_exhausted_keys(key_manager):
    """Test that get_key skips exhausted keys."""
    model = "gemini-1.5-flash"

    # Exhaust key_1 by marking it
    await key_manager.update_usage(
        model_name=model,
        key_value="test_key_1",
        is_vertex_key=False,
        tokens_used=9999999999,
        error=True,
        error_type="429",
    )

    # Get key should not return key_1
    for _ in range(10):
        key = await key_manager.get_key(model, is_vertex_key=False)
        assert key != "test_key_1"


@pytest.mark.asyncio
async def test_get_key_skips_inactive_keys(key_manager):
    """Test that get_key skips inactive keys."""
    model = "gemini-1.5-flash"

    # Deactivate key_1
    await key_manager.update_usage(
        model_name=model,
        key_value="test_key_1",
        is_vertex_key=False,
        tokens_used=1000,
        error=True,
        error_type="permanent",
    )

    # Get key should not return key_1
    for _ in range(10):
        key = await key_manager.get_key(model, is_vertex_key=False)
        assert key != "test_key_1"


@pytest.mark.asyncio
async def test_get_key_unknown_model_fallback(key_manager):
    """Test that get_key falls back to cycling for unknown models."""
    key = await key_manager.get_key("unknown-model", is_vertex_key=False)
    assert key in key_manager.api_keys


@pytest.mark.asyncio
async def test_get_key_no_available_keys_raises_exception(key_manager):
    """Test that get_key raises exception when no keys are available."""
    model = "gemini-1.5-flash"

    # Exhaust all keys by setting them as exhausted
    async with key_manager.lock.write_lock():
        for key_val in key_manager.api_keys:
            idx = (model, False, key_val)
            key_manager.df.loc[idx, "is_exhausted"] = True
        # Update available usage to ensure left values are correct
        await key_manager._set_available_usage()

    key = await key_manager.get_key(model, is_vertex_key=False)
    assert key == ""


# ============================================================================
# Update Usage Tests
# ============================================================================


@pytest.mark.asyncio
async def test_update_usage_increments_counters(key_manager):
    """Test that update_usage correctly increments usage counters."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Get initial values
    async with key_manager.lock.read_lock():
        idx = (model, False, key)
        initial_rpm = int(key_manager.df.loc[idx, "rpm"])
        initial_tpm = int(key_manager.df.loc[idx, "tpm"])
        initial_rpd = int(key_manager.df.loc[idx, "rpd"])

    # Update usage
    tokens = 1000
    await key_manager.update_usage(
        model_name=model,
        key_value=key,
        is_vertex_key=False,
        tokens_used=tokens,
    )

    # Check updated values
    async with key_manager.lock.read_lock():
        new_rpm = int(key_manager.df.loc[idx, "rpm"])
        new_tpm = int(key_manager.df.loc[idx, "tpm"])
        new_rpd = int(key_manager.df.loc[idx, "rpd"])

    assert new_rpm == initial_rpm + 1
    assert new_tpm == initial_tpm + tokens
    assert new_rpd == initial_rpd + 1


@pytest.mark.asyncio
async def test_update_usage_updates_last_used(key_manager):
    """Test that update_usage updates the last_used timestamp."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Get initial timestamp
    async with key_manager.lock.read_lock():
        idx = (model, False, key)
        initial_last_used = key_manager.df.loc[idx, "last_used"]

    # Wait a bit
    await asyncio.sleep(0.1)

    # Update usage
    await key_manager.update_usage(
        model_name=model,
        key_value=key,
        is_vertex_key=False,
        tokens_used=100,
    )

    # Check updated timestamp
    async with key_manager.lock.read_lock():
        new_last_used = key_manager.df.loc[idx, "last_used"]

    assert new_last_used > initial_last_used


@pytest.mark.asyncio
async def test_update_usage_with_permanent_error_deactivates_key(key_manager):
    """Test that permanent errors deactivate the key for all models."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Trigger permanent error
    await key_manager.update_usage(
        model_name=model,
        key_value=key,
        is_vertex_key=False,
        tokens_used=0,
        error=True,
        error_type="permanent",
    )

    # Check that key is deactivated for ALL models
    async with key_manager.lock.read_lock():
        for model_name in key_manager.rate_limit_data.keys():
            idx = (model_name, False, key)
            assert bool(key_manager.df.loc[idx, "is_active"]) is False


@pytest.mark.asyncio
async def test_update_usage_with_429_error_exhausts_model(key_manager):
    """Test that 429 errors exhaust only the specific model."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Trigger 429 error
    await key_manager.update_usage(
        model_name=model,
        key_value=key,
        is_vertex_key=False,
        tokens_used=0,
        error=True,
        error_type="429",
    )

    # Check that only this model is exhausted
    async with key_manager.lock.read_lock():
        idx = (model, False, key)
        assert bool(key_manager.df.loc[idx, "is_exhausted"]) is True

        # Other models should not be exhausted
        for other_model in key_manager.rate_limit_data.keys():
            if other_model != model:
                other_idx = (other_model, False, key)
                assert bool(key_manager.df.loc[other_idx, "is_exhausted"]) is False


@pytest.mark.asyncio
async def test_update_usage_unknown_model(key_manager):
    """Test update_usage with unknown model."""
    result = await key_manager.update_usage(
        model_name="unknown-model",
        key_value="test_key_1",
        is_vertex_key=False,
        tokens_used=100,
    )
    # Should return True even for unknown models (they're just not tracked)
    assert result is True


# ============================================================================
# Reset Usage Tests
# ============================================================================


@pytest.mark.asyncio
async def test_reset_usage_clears_minute_metrics(key_manager):
    """Test that reset_usage clears minute-level metrics."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Add some usage
    await key_manager.update_usage(
        model_name=model,
        key_value=key,
        is_vertex_key=False,
        tokens_used=1000,
    )

    # Force a minute reset by manipulating timestamps
    async with key_manager.lock.write_lock():
        key_manager.last_minute_reset_ts = key_manager.now_minute() - timedelta(
            minutes=2
        )

    # Run reset
    result = await key_manager.reset_usage()
    assert result is True

    # Check that rpm and tpm are reset
    async with key_manager.lock.read_lock():
        idx = (model, False, key)
        assert key_manager.df.loc[idx, "rpm"] == 0
        assert key_manager.df.loc[idx, "tpm"] == 0
        assert bool(key_manager.df.loc[idx, "is_exhausted"]) is False


@pytest.mark.asyncio
async def test_reset_usage_clears_day_metrics(key_manager):
    """Test that reset_usage clears day-level metrics."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Add some usage
    await key_manager.update_usage(
        model_name=model,
        key_value=key,
        is_vertex_key=False,
        tokens_used=1000,
    )

    # Force a day reset
    async with key_manager.lock.write_lock():
        key_manager.last_day_reset_ts = key_manager.now_day() - timedelta(days=2)

    # Run reset
    result = await key_manager.reset_usage()
    assert result is True

    # Check that rpd is reset
    async with key_manager.lock.read_lock():
        idx = (model, False, key)
        assert key_manager.df.loc[idx, "rpd"] == 0


@pytest.mark.asyncio
async def test_reset_usage_does_not_reset_prematurely(key_manager):
    """Test that reset_usage doesn't reset before the interval."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Add some usage
    await key_manager.update_usage(
        model_name=model,
        key_value=key,
        is_vertex_key=False,
        tokens_used=1000,
    )

    async with key_manager.lock.read_lock():
        idx = (model, False, key)
        rpm_before = int(key_manager.df.loc[idx, "rpm"])

    # Run reset immediately (should not reset)
    await key_manager.reset_usage()

    async with key_manager.lock.read_lock():
        rpm_after = int(key_manager.df.loc[idx, "rpm"])

    # Values should be unchanged
    assert rpm_after == rpm_before


# ============================================================================
# Exhaustion Tests
# ============================================================================


@pytest.mark.asyncio
async def test_exhaustion_when_rpm_limit_reached(key_manager):
    """Test that keys are marked exhausted when RPM limit is reached."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Get max RPM
    async with key_manager.lock.read_lock():
        idx = (model, False, key)
        max_rpm = int(key_manager.df.loc[idx, "max_rpm"])

    # Use up to the limit
    for _ in range(max_rpm):
        await key_manager.update_usage(
            model_name=model,
            key_value=key,
            is_vertex_key=False,
            tokens_used=100,
        )

    # Check exhaustion
    async with key_manager.lock.read_lock():
        assert bool(key_manager.df.loc[idx, "is_exhausted"]) is True


@pytest.mark.asyncio
async def test_exhaustion_when_tpm_limit_reached(key_manager):
    """Test that keys are marked exhausted when TPM limit is reached."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Get max TPM
    async with key_manager.lock.read_lock():
        idx = (model, False, key)
        max_tpm = int(key_manager.df.loc[idx, "max_tpm"])

    # Use up to the limit in one call
    await key_manager.update_usage(
        model_name=model,
        key_value=key,
        is_vertex_key=False,
        tokens_used=max_tpm,
    )

    # Check exhaustion
    async with key_manager.lock.read_lock():
        assert bool(key_manager.df.loc[idx, "is_exhausted"]) is True


@pytest.mark.asyncio
async def test_exhaustion_when_rpd_limit_reached(key_manager):
    """Test that keys are marked exhausted when RPD limit is reached."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Get max RPD
    async with key_manager.lock.read_lock():
        idx = (model, False, key)
        max_rpd = int(key_manager.df.loc[idx, "max_rpd"])

    # Use up to the limit
    for _ in range(max_rpd):
        await key_manager.update_usage(
            model_name=model,
            key_value=key,
            is_vertex_key=False,
            tokens_used=100,
        )

    # Check exhaustion
    async with key_manager.lock.read_lock():
        assert bool(key_manager.df.loc[idx, "is_exhausted"]) is True


# ============================================================================
# Database Persistence Tests
# ============================================================================


@pytest.mark.asyncio
async def test_commit_to_db_saves_state(key_manager):
    """Test that _commit_to_db saves state to database."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Add some usage
    await key_manager.update_usage(
        model_name=model,
        key_value=key,
        is_vertex_key=False,
        tokens_used=5000,
    )

    # Verify DataFrame has the updated values before committing
    async with key_manager.lock.read_lock():
        idx = (model, False, key)
        df_rpm = int(key_manager.df.loc[idx, "rpm"])
        df_tpm = int(key_manager.df.loc[idx, "tpm"])
        df_rpd = int(key_manager.df.loc[idx, "rpd"])
        assert df_rpm >= 1, f"DataFrame rpm should be >= 1, got {df_rpm}"
        assert df_tpm >= 5000, f"DataFrame tpm should be >= 5000, got {df_tpm}"
        assert df_rpd >= 1, f"DataFrame rpd should be >= 1, got {df_rpd}"

    # Commit to DB - this should save the DataFrame values
    await key_manager._commit_to_db()

    # Wait a bit longer to ensure commit completes and background task doesn't interfere
    # The background task runs every 1 second, so we wait 1.5 seconds to be safe
    await asyncio.sleep(1.5)

    # Query database directly with a fresh session
    async with key_manager.db_maker() as session:
        from sqlalchemy import select

        stmt = select(UsageMatrix).where(
            UsageMatrix.api_key == key,
            UsageMatrix.model_name == model,
            UsageMatrix.vertex_key == False,  # noqa: E712
        )
        result = await session.execute(stmt)
        record = result.scalars().first()

        assert record is not None
        # Verify the values were actually saved
        assert int(record.rpm) >= 1, f"Expected rpm >= 1, got {record.rpm}"
        assert int(record.tpm) >= 5000, f"Expected tpm >= 5000, got {record.tpm}"
        assert int(record.rpd) >= 1, f"Expected rpd >= 1, got {record.rpd}"


@pytest.mark.asyncio
async def test_load_from_db_restores_state(
    sample_api_keys,
    sample_vertex_keys,
    test_session_maker,
    mock_rate_limit_data,
):
    """Test that _load_from_db restores state from database."""
    # Create first instance and add usage
    with patch(
        "app.service.key.key_manager_v2.scrape_gemini_rate_limits",
        return_value={"Free Tier": mock_rate_limit_data},
    ):
        km1 = KeyManager(
            api_keys=sample_api_keys,
            vertex_api_keys=sample_vertex_keys,
            async_session_maker=test_session_maker,
            rate_limit_data=mock_rate_limit_data,
        )
        await km1.init()

        model = "gemini-1.5-flash"
        key = "test_key_1"

        await km1.update_usage(
            model_name=model,
            key_value=key,
            is_vertex_key=False,
            tokens_used=5000,
        )

        # Get usage values
        async with km1.lock.read_lock():
            idx = (model, False, key)
            rpm_saved = int(km1.df.loc[idx, "rpm"])  # type: ignore
            tpm_saved = int(km1.df.loc[idx, "tpm"])  # type: ignore

        await km1.shutdown()

    # Create new instance - should load from DB
    with patch(
        "app.service.key.key_manager_v2.scrape_gemini_rate_limits",
        return_value={"Free Tier": mock_rate_limit_data},
    ):
        km2 = KeyManager(
            api_keys=sample_api_keys,
            vertex_api_keys=sample_vertex_keys,
            async_session_maker=test_session_maker,
            rate_limit_data=mock_rate_limit_data,
        )
        await km2.init()

        # Check restored values
        async with km2.lock.read_lock():
            idx = (model, False, key)
            rpm_loaded = int(km2.df.loc[idx, "rpm"])  # type: ignore
            tpm_loaded = int(km2.df.loc[idx, "tpm"])  # type: ignore

        assert rpm_loaded == rpm_saved
        assert tpm_loaded == tpm_saved

        await km2.shutdown()


# ============================================================================
# Background Task Tests
# ============================================================================


@pytest.mark.asyncio
async def test_background_task_starts_on_init(
    sample_api_keys,
    sample_vertex_keys,
    test_session_maker,
    mock_rate_limit_data,
):
    """Test that background task starts on initialization."""
    with patch(
        "app.service.key.key_manager_v2.scrape_gemini_rate_limits",
        return_value={"Free Tier": mock_rate_limit_data},
    ):
        km = KeyManager(
            api_keys=sample_api_keys,
            vertex_api_keys=sample_vertex_keys,
            async_session_maker=test_session_maker,
            rate_limit_data=mock_rate_limit_data,
        )
        await km.init()

        assert km._background_task is not None
        assert not km._background_task.done()

        await km.shutdown()
        assert km._background_task.done()


@pytest.mark.asyncio
async def test_background_task_performs_resets(key_manager):
    """Test that background task performs periodic resets."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Add usage
    await key_manager.update_usage(
        model_name=model,
        key_value=key,
        is_vertex_key=False,
        tokens_used=1000,
    )

    # Force a reset by manipulating timestamp
    async with key_manager.lock.write_lock():
        idx = (model, False, key)
        rpm_before = int(key_manager.df.loc[idx, "rpm"])
        key_manager.last_minute_reset_ts = key_manager.now_minute() - timedelta(
            minutes=2
        )

    assert rpm_before > 0

    # Wait for background task to run
    await asyncio.sleep(2)

    # Check that reset happened
    async with key_manager.lock.read_lock():
        rpm_after = int(key_manager.df.loc[idx, "rpm"])

    assert rpm_after == 0


# ============================================================================
# Shutdown Tests
# ============================================================================


@pytest.mark.asyncio
async def test_shutdown_stops_background_task(key_manager):
    """Test that shutdown stops the background task."""
    assert key_manager._background_task is not None
    assert not key_manager._background_task.done()

    await key_manager.shutdown()

    assert key_manager._background_task.done()
    assert key_manager._stop_event.is_set()
    assert key_manager.is_ready is False


@pytest.mark.asyncio
async def test_shutdown_commits_to_db(key_manager):
    """Test that shutdown performs a final DB commit."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Add usage
    await key_manager.update_usage(
        model_name=model,
        key_value=key,
        is_vertex_key=False,
        tokens_used=7500,
    )

    # Shutdown
    await key_manager.shutdown()

    # Verify data is in DB
    async with key_manager.db_maker() as session:
        from sqlalchemy import select

        stmt = select(UsageMatrix).where(
            UsageMatrix.api_key == key,
            UsageMatrix.model_name == model,
        )
        result = await session.execute(stmt)
        record = result.scalars().first()

        assert record is not None
        assert record.tpm >= 7500


@pytest.mark.asyncio
async def test_shutdown_idempotent(key_manager):
    """Test that shutdown can be called multiple times safely."""
    await key_manager.shutdown()
    # Should not raise error
    await key_manager.shutdown()
    await key_manager.shutdown()


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_get_key_calls(key_manager):
    """Test concurrent get_key calls are thread-safe."""
    model = "gemini-1.5-flash"

    async def get_key_task():
        return await key_manager.get_key(model, is_vertex_key=False)

    # Run multiple concurrent get_key calls
    tasks = [get_key_task() for _ in range(10)]
    keys = await asyncio.gather(*tasks)

    # All should return valid keys
    assert all(k in key_manager.api_keys for k in keys)


@pytest.mark.asyncio
async def test_concurrent_update_usage_calls(key_manager):
    """Test concurrent update_usage calls are thread-safe."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    async def update_task():
        return await key_manager.update_usage(
            model_name=model,
            key_value=key,
            is_vertex_key=False,
            tokens_used=100,
        )

    # Run multiple concurrent updates
    tasks = [update_task() for _ in range(20)]
    results = await asyncio.gather(*tasks)

    # All should succeed
    assert all(results)

    # RPM should be exactly 20
    async with key_manager.lock.read_lock():
        idx = (model, False, key)
        rpm = int(key_manager.df.loc[idx, "rpm"])

    assert rpm == 20


@pytest.mark.asyncio
async def test_check_ready_validates_dataframe(key_manager):
    """Test that _check_ready validates DataFrame structure."""
    # Should pass when properly initialized
    assert await key_manager._check_ready()

    # Break the DataFrame
    async with key_manager.lock.write_lock():
        key_manager.df = pd.DataFrame()

    # Should fail
    with pytest.raises(Exception):
        await key_manager._check_ready()


@pytest.mark.asyncio
async def test_ensure_numeric_columns(key_manager):
    """Test that _ensure_numeric_columns converts types properly."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Corrupt data types and fix within the same lock
    async with key_manager.lock.write_lock():
        idx = (model, False, key)
        key_manager.df.loc[idx, "rpm"] = "not_a_number"
        # Fix it within the lock
        await key_manager._ensure_numeric_columns()
        rpm = key_manager.df.loc[idx, "rpm"]
        # Check for both Python and numpy numeric types
        import numpy as np

        assert isinstance(rpm, (int, float, np.integer, np.floating))


@pytest.mark.asyncio
async def test_set_available_usage_clips_negative_values(key_manager):
    """Test that available usage never goes negative."""
    model = "gemini-1.5-flash"
    key = "test_key_1"

    # Manually set usage above max and recalculate within the same lock
    async with key_manager.lock.write_lock():
        idx = (model, False, key)
        max_rpm = int(key_manager.df.loc[idx, "max_rpm"])
        key_manager.df.loc[idx, "rpm"] = max_rpm + 100
        # Recalculate available usage within the lock
        await key_manager._set_available_usage()
        # Should be clipped to 0, not negative
        rpm_left = int(key_manager.df.loc[idx, "rpm_left"])
        assert rpm_left == 0


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_full_lifecycle_integration(
    sample_api_keys,
    sample_vertex_keys,
    test_session_maker,
    mock_rate_limit_data,
):
    """Test a full lifecycle: init -> use -> reset -> shutdown -> reload."""
    with patch(
        "app.service.key.key_manager_v2.scrape_gemini_rate_limits",
        return_value={"Free Tier": mock_rate_limit_data},
    ):
        # Phase 1: Initialize and use
        km1 = KeyManager(
            api_keys=sample_api_keys,
            vertex_api_keys=sample_vertex_keys,
            async_session_maker=test_session_maker,
            rate_limit_data=mock_rate_limit_data,
        )
        await km1.init()

        model = "gemini-1.5-flash"

        # Get and use keys
        for _ in range(5):
            key = await km1.get_key(model, is_vertex_key=False)
            await km1.update_usage(
                model_name=model,
                key_value=key,
                is_vertex_key=False,
                tokens_used=1000,
            )

        # Verify usage - need to filter by model and is_vertex_key
        async with km1.lock.read_lock():
            model_df = km1.df.xs(model, level="model_name")
            non_vertex_df = model_df.xs(False, level="is_vertex_key")
            total_rpm = non_vertex_df["rpm"].sum()
            assert total_rpm == 5

        await km1.shutdown()

        # Phase 2: Reload from database
        km2 = KeyManager(
            api_keys=sample_api_keys,
            vertex_api_keys=sample_vertex_keys,
            async_session_maker=test_session_maker,
            rate_limit_data=mock_rate_limit_data,
        )
        await km2.init()

        # Verify state was restored
        async with km2.lock.read_lock():
            model_df = km2.df.xs(model, level="model_name")
            non_vertex_df = model_df.xs(False, level="is_vertex_key")
            total_rpm = non_vertex_df["rpm"].sum()
            assert total_rpm == 5

        await km2.shutdown()


@pytest.mark.asyncio
async def test_rate_limiting_behavior(key_manager):
    """Test that rate limiting prevents over-usage."""
    model = "gemini-1.5-flash"

    # Get max RPM for this model
    async with key_manager.lock.read_lock():
        sample_idx = (model, False, key_manager.api_keys[0])
        max_rpm = int(key_manager.df.loc[sample_idx, "max_rpm"])

    total_keys = len(key_manager.api_keys)
    total_available_requests = max_rpm * total_keys

    # Try to make more requests than available
    successful_requests = 0
    for _ in range(total_available_requests + 10):
        try:
            key = await key_manager.get_key(model, is_vertex_key=False)
            # If get_key returns empty string, no keys are available
            if not key:
                break
            await key_manager.update_usage(
                model_name=model,
                key_value=key,
                is_vertex_key=False,
                tokens_used=100,
            )
            successful_requests += 1
        except Exception:
            # Expected to fail when exhausted
            break

    # Should have made exactly the available number of requests
    assert successful_requests == total_available_requests
