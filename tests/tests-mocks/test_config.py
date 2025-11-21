import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.config.config import Settings, _parse_db_value, sync_initial_settings, CommaSeparatedListEnvSettingsSource
from typing import List, Dict

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("DATABASE_TYPE", "sqlite")
    monkeypatch.setenv("API_KEYS", "key1,key2,key3")

def test_settings_initialization_with_custom_source(mock_env_vars):
    """Test that the Settings class is initialized correctly with the custom source."""
    settings = Settings()
    assert settings.API_KEYS == ["key1", "key2", "key3"]

@pytest.mark.asyncio
async def test_sync_initial_settings(mock_env_vars):
    """Test the sync_initial_settings function."""
    # Mock AsyncSessionLocal to return a mock session
    mock_session = AsyncMock()
    mock_result = MagicMock()
    # Mock the query result - returns rows with key and value
    mock_row = MagicMock()
    mock_row._mapping = {"key": "API_KEYS", "value": '["key4","key5"]'}
    mock_result.scalars.return_value.all.return_value = [mock_row]
    
    # Mock all execute calls to return the initial result for the first call,
    # and empty results for subsequent calls (sync-back queries)
    execute_call_count = [0]
    async def mock_execute(query):
        execute_call_count[0] += 1
        if execute_call_count[0] == 1:
            # First call: initial select query
            return mock_result
        # Subsequent calls: return empty result for description queries and inserts/updates
        empty_result = MagicMock()
        empty_result.fetchall.return_value = []
        return empty_result
    
    mock_session.execute = AsyncMock(side_effect=mock_execute)
    mock_session.begin.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    
    # Fix: Patch where AsyncSessionLocal is actually defined
    with patch("app.database.connection.AsyncSessionLocal", return_value=mock_context_manager):
        await sync_initial_settings()
        from app.config.config import settings
        assert settings.API_KEYS == ["key4", "key5"]

def test_parse_db_value():
    """Test the _parse_db_value function."""
    assert _parse_db_value("key", "val1,val2", List[str]) == ["val1", "val2"]
    assert _parse_db_value("key", '["val1","val2"]', List[str]) == ["val1", "val2"]
    assert _parse_db_value("key", '{"k":"v"}', Dict[str, str]) == {"k": "v"}
    assert _parse_db_value("key", "true", bool) is True
    assert _parse_db_value("key", "123", int) == 123
    assert _parse_db_value("key", "123.45", float) == 123.45
    assert _parse_db_value("key", "astring", str) == "astring"

def test_comma_separated_list_env_settings_source():
    """Test the CommaSeparatedListEnvSettingsSource."""
    field = MagicMock()
    field.annotation = List[str]
    source = CommaSeparatedListEnvSettingsSource(Settings)
    value = source.prepare_field_value("API_KEYS", field, "key1,key2,key3", False)
    assert value == ["key1", "key2", "key3"]
