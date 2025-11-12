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
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.service.key.key_manager_v2 import (
    KeyManager,
    UsageMatrix,
    Base,
    init_database,
    get_db,
)


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
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest_asyncio.fixture
async def test_session_maker(test_engine):
    """Create a session maker for testing."""
    return async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
def mock_rate_limit_data():
    """Mock rate limit data for testing."""
    return {
        "gemini-1.5-flash": {"RPM": 15, "TPM": 1000000, "RPD": 1500},
        "gemini-1.5-pro": {"RPM": 10, "TPM": 500000, "RPD": 1000},
        "gemini-2.0-flash": {"RPM": 20, "TPM": 2000000, "RPD": 2000},
    }


@pytest_asyncio.fixture
def sample_api_keys():
    """Sample API keys for testing."""
    return ["test_key_1", "test_key_2", "test_key_3"]


@pytest_asyncio.fixture
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
    km = KeyManager(
        api_keys=sample_api_keys,
        vertex_api_keys=sample_vertex_keys,
        async_session_maker=test_session_maker,
        rate_limit_data=mock_rate_limit_data,
        minute_reset_interval=60,
    )
    
    # Mock the rate limit scraping
    with patch(
        "app.service.key.key_manager_v2.scrape_gemini_rate_limits",
        return_value={"Free Tier": mock_rate_limit_data},
    ):
        await km.init()
    
    yield km
    
    # Cleanup
    await km.shutdown()


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.asyncio
async def test_init_database(test_engine):
    """Test database initialization."""
    async with test_engine.begin() as conn:
        # Verify tables are created
        result = await conn.run_sync(
            lambda sync_conn: sync_conn.dialect.has_table(sync_conn, "t_usage_stats")
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
        "rpm", "tpm", "rpd",
        "max_rpm", "max_tpm", "max_rpd",
        "minute_reset_time", "day_reset_time",
        "is_active", "is_exhausted", "last_used",
        "rpm_left", "tpm_left", "rpd_left",
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
    matched, normalized = key_manager._model_normalization("gemini-1.5-flash")
    assert matched is True
    assert normalized == "gemini-1.5-flash"


@pytest.mark.asyncio
async def test_model_normalization_prefix_match(key_manager):
    """Test model normalization with prefix match."""
    matched, normalized = key_manager._model_normalization("gemini-1.5-flash-001")
    assert matched is True
    assert normalized == "gemini-1.5-flash"


@pytest.mark.asyncio
async def test_model_normalization_no_match(key_manager):
    """Test model normalization with no match."""
    matched, normalized = key_manager._model_normalization("unknown-model")
    assert matched is False
    assert normalized == "unknown-model"


@pytest.mark.asyncio
async def test_model_normalization_longest_prefix(key_manager):
    """Test that model normalization returns the longest matching prefix."""
    # If we have both "gemini-1.5" and "gemini-1.5-flash" in rate limits,
    # it should match "gemini-1.5-flash" for "gemini-1.5-flash-001"
    matched, normalized = key_manager._model_normalization("gemini-1.5-flash-search")
    assert matched is True
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
async def test_get_key_returns_valid_key(key_manager):
    """Test that get_key returns a valid API key."""
    key = await key_manager.get_key("gemini-1.5-flash", is_vertex_key=False)
    assert key in key_manager.api_keys


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
    async with key_manager.lock:
        idx = (model, False, "test_key_1")
        key_manager.df.loc[idx, "is_exhausted"] = True
        await key_manager._set_available_usage()
    
    # Get key should not return key_1
    for _ in range(10):
        key = await key_manager.get_key(model, is_vertex_key=False)
        assert key != "test_key_1"


@pytest.mark.asyncio
async def test_get_key_skips_inactive_keys(key_manager):
    """Test that get_key skips inactive keys."""
    model = "gemini-1.5-flash"
    
    # Deactivate key_1
    async with key_manager.lock:
        idx = (model, False, "test_key_1")
        key_manager.df.loc[idx, "is_active"] = False
    
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
    
    # Exhaust all keys
    async with key_manager.lock:
        for key_val in key_manager.api_keys:
            idx = (model, False, key_val)
            key_manager.df.loc[idx, "is_exhausted"] = True
        await key_manager._set_available_usage()
    
    with pytest.raises(Exception, match="No available keys"):
        await key_manager.get_key(model, is_vertex_key=False)


# ============================================================================
# Update Usage Tests
# ============================================================================


@pytest.mark.asyncio
async def test_update_usage_increments_counters(key_manager):
    """Test that update_usage correctly increments usage counters."""
    model = "gemini-1.5-flash"
    key = "test_key_1"
    
    # Get initial values
    async with key_manager.lock:
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
    async with key_manager.lock:
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
    async with key_manager.lock:
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
    async with key_manager.lock:
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
    async with key_manager.lock:
        for model_name in key_manager.rate_limit_data.keys():
            idx = (model_name, False, key)
            assert key_manager.df.loc[idx, "is_active"] is False


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
    async with key_manager.lock:
        idx = (model, False, key)
        assert key_manager.df.loc[idx, "is_exhausted"] is True
        
        # Other models should not be exhausted
        for other_model in key_manager.rate_limit_data.keys():
            if other_model != model:
                other_idx = (other_model, False, key)
                assert key_manager.df.loc[other_idx, "is_exhausted"] is False


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
    async with key_manager.lock:
        key_manager.last_minute_reset_ts = key_manager.now_minute() - timedelta(minutes=2)
    
    # Run reset
    result = await key_manager.reset_usage()
    assert result is True
    
    # Check that rpm and tpm are reset
    async with key_manager.lock:
        idx = (model, False, key)
        assert key_manager.df.loc[idx, "rpm"] == 0
        assert key_manager.df.loc[idx, "tpm"] == 0
        assert key_manager.df.loc[idx, "is_exhausted"] is False


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
    async with key_manager.lock:
        key_manager.last_day_reset_ts = key_manager.now_day() - timedelta(days=2)
    
    # Run reset
    result = await key_manager.reset_usage()
    assert result is True
    
    # Check that rpd is reset
    async with key_manager.lock:
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
    
    async with key_manager.lock:
        idx = (model, False, key)
        rpm_before = int(key_manager.df.loc[idx, "rpm"])
    
    # Run reset immediately (should not reset)
    await key_manager.reset_usage()
    
    async with key_manager.lock:
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
    async with key_manager.lock:
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
    async with key_manager.lock:
        assert key_manager.df.loc[idx, "is_exhausted"] is True


@pytest.mark.asyncio
async def test_exhaustion_when_tpm_limit_reached(key_manager):
    """Test that keys are marked exhausted when TPM limit is reached."""
    model = "gemini-1.5-flash"
    key = "test_key_1"
    
    # Get max TPM
    async with key_manager.lock:
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
    async with key_manager.lock:
        assert key_manager.df.loc[idx, "is_exhausted"] is True


@pytest.mark.asyncio
async def test_exhaustion_when_rpd_limit_reached(key_manager):
    """Test that keys are marked exhausted when RPD limit is reached."""
    model = "gemini-1.5-flash"
    key = "test_key_1"
    
    # Get max RPD
    async with key_manager.lock:
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
    async with key_manager.lock:
        assert key_manager.df.loc[idx, "is_exhausted"] is True


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
    
    # Commit to DB
    await key_manager._commit_to_db()
    
    # Query database directly
    async with key_manager.db_maker() as session:
        from sqlalchemy import select
        stmt = select(UsageMatrix).where(
            UsageMatrix.api_key == key,
            UsageMatrix.model_name == model,
        )
        result = await session.execute(stmt)
        record = result.scalars().first()
        
        assert record is not None
        assert record.rpm >= 1
        assert record.tpm >= 5000
        assert record.rpd >= 1


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
        async with km1.lock:
            idx = (model, False, key)
            rpm_saved = int(km1.df.loc[idx, "rpm"])
            tpm_saved = int(km1.df.loc[idx, "tpm"])
        
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
        async with km2.lock:
            idx = (model, False, key)
            rpm_loaded = int(km2.df.loc[idx, "rpm"])
            tpm_loaded = int(km2.df.loc[idx, "tpm"])
        
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
    async with key_manager.lock:
        idx = (model, False, key)
        rpm_before = int(key_manager.df.loc[idx, "rpm"])
        key_manager.last_minute_reset_ts = key_manager.now_minute() - timedelta(minutes=2)
    
    assert rpm_before > 0
    
    # Wait for background task to run
    await asyncio.sleep(2)
    
    # Check that reset happened
    async with key_manager.lock:
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
    async with key_manager.lock:
        idx = (model, False, key)
        rpm = int(key_manager.df.loc[idx, "rpm"])
    
    assert rpm == 20


@pytest.mark.asyncio
async def test_check_ready_validates_dataframe(key_manager):
    """Test that _check_ready validates DataFrame structure."""
    # Should pass when properly initialized
    assert await key_manager._check_ready()
    
    # Break the DataFrame
    async with key_manager.lock:
        key_manager.df = pd.DataFrame()
    
    # Should fail
    with pytest.raises(Exception):
        await key_manager._check_ready()


@pytest.mark.asyncio
async def test_ensure_numeric_columns(key_manager):
    """Test that _ensure_numeric_columns converts types properly."""
    model = "gemini-1.5-flash"
    key = "test_key_1"
    
    # Corrupt data types
    async with key_manager.lock:
        idx = (model, False, key)
        key_manager.df.loc[idx, "rpm"] = "not_a_number"
    
    # Should fix it
    await key_manager._ensure_numeric_columns()
    
    async with key_manager.lock:
        rpm = key_manager.df.loc[idx, "rpm"]
        assert isinstance(rpm, (int, float))


@pytest.mark.asyncio
async def test_set_available_usage_clips_negative_values(key_manager):
    """Test that available usage never goes negative."""
    model = "gemini-1.5-flash"
    key = "test_key_1"
    
    # Manually set usage above max
    async with key_manager.lock:
        idx = (model, False, key)
        max_rpm = int(key_manager.df.loc[idx, "max_rpm"])
        key_manager.df.loc[idx, "rpm"] = max_rpm + 100
    
    # Recalculate available usage
    await key_manager._set_available_usage()
    
    # Should be clipped to 0, not negative
    async with key_manager.lock:
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
        
        # Verify usage
        async with km1.lock:
            total_rpm = km1.df.loc[(model, False), "rpm"].sum()
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
        async with km2.lock:
            total_rpm = km2.df.loc[(model, False), "rpm"].sum()
            assert total_rpm == 5
        
        await km2.shutdown()


@pytest.mark.asyncio
async def test_rate_limiting_behavior(key_manager):
    """Test that rate limiting prevents over-usage."""
    model = "gemini-1.5-flash"
    
    # Get max RPM for this model
    async with key_manager.lock:
        sample_idx = (model, False, key_manager.api_keys[0])
        max_rpm = int(key_manager.df.loc[sample_idx, "max_rpm"])
    
    total_keys = len(key_manager.api_keys)
    total_available_requests = max_rpm * total_keys
    
    # Try to make more requests than available
    successful_requests = 0
    for _ in range(total_available_requests + 10):
        try:
            key = await key_manager.get_key(model, is_vertex_key=False)
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

