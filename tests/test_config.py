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
    with patch("app.database.connection.database", new_callable=AsyncMock) as mock_database:
        mock_database.is_connected = True
        mock_database.fetch_all.return_value = [{"key": "API_KEYS", "value": '["key4","key5"]'}]
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
