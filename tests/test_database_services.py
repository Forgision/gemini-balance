import pytest
from unittest.mock import AsyncMock, patch
from app.database import services
from app.database.models import FileState

@pytest.fixture
def mock_database():
    """Fixture for mocking the database object."""
    with patch("app.database.services.database", new_callable=AsyncMock) as mock_db:
        yield mock_db

@pytest.mark.asyncio
async def test_get_all_settings(mock_database):
    """Test the get_all_settings function."""
    mock_database.fetch_all.return_value = []
    await services.get_all_settings()
    mock_database.fetch_all.assert_called_once()

@pytest.mark.asyncio
async def test_get_setting(mock_database):
    """Test the get_setting function."""
    mock_database.fetch_one.return_value = None
    await services.get_setting("some_key")
    mock_database.fetch_one.assert_called_once()

@pytest.mark.asyncio
async def test_update_setting(mock_database):
    """Test the update_setting function."""
    with patch("app.database.services.get_setting", new_callable=AsyncMock) as mock_get_setting:
        mock_get_setting.return_value = {"key": "some_key", "value": "old_value", "description": ""}
        await services.update_setting("some_key", "new_value")
        mock_database.execute.assert_called_once()

@pytest.mark.asyncio
async def test_add_error_log(mock_database):
    """Test the add_error_log function."""
    await services.add_error_log("gemini_key", "model_name", "error_type", "error_log")
    mock_database.execute.assert_called_once()

@pytest.mark.asyncio
async def test_get_error_logs(mock_database):
    """Test the get_error_logs function."""
    mock_database.fetch_all.return_value = []
    await services.get_error_logs()
    mock_database.fetch_all.assert_called_once()

@pytest.mark.asyncio
async def test_delete_error_logs_by_ids(mock_database):
    """Test the delete_error_logs_by_ids function."""
    await services.delete_error_logs_by_ids([1, 2, 3])
    mock_database.execute.assert_called_once()

@pytest.mark.asyncio
async def test_create_file_record(mock_database):
    """Test the create_file_record function."""
    with patch("app.database.services.get_file_record_by_name", new_callable=AsyncMock) as mock_get_file_record_by_name:
        mock_get_file_record_by_name.return_value = {"name": "files/file_id"}
        await services.create_file_record("files/file_id", "mime/type", 123, "api_key", "uri", "2024-01-01", "2024-01-01", "2025-01-01")
        mock_database.execute.assert_called_once()

@pytest.mark.asyncio
async def test_get_file_record_by_name(mock_database):
    """Test the get_file_record_by_name function."""
    mock_database.fetch_one.return_value = None
    await services.get_file_record_by_name("files/file_id")
    mock_database.fetch_one.assert_called_once()

@pytest.mark.asyncio
async def test_update_file_record_state(mock_database):
    """Test the update_file_record_state function."""
    mock_database.execute.return_value = 1
    await services.update_file_record_state("files/file_id", FileState.ACTIVE)
    mock_database.execute.assert_called_once()

@pytest.mark.asyncio
async def test_list_file_records(mock_database):
    """Test the list_file_records function."""
    mock_database.fetch_all.return_value = []
    await services.list_file_records()
    mock_database.fetch_all.assert_called_once()

@pytest.mark.asyncio
async def test_delete_file_record(mock_database):
    """Test the delete_file_record function."""
    await services.delete_file_record("files/file_id")
    mock_database.execute.assert_called_once()

@pytest.mark.asyncio
async def test_delete_expired_file_records(mock_database):
    """Test the delete_expired_file_records function."""
    mock_database.fetch_all.return_value = []
    await services.delete_expired_file_records()
    mock_database.fetch_all.assert_called_once()

@pytest.mark.asyncio
async def test_get_usage_stats_by_key_and_model(mock_database):
    """Test the get_usage_stats_by_key_and_model function."""
    mock_database.fetch_one.return_value = None
    await services.get_usage_stats_by_key_and_model("api_key", "model_name")
    mock_database.fetch_one.assert_called_once()

@pytest.mark.asyncio
async def test_get_all_usage_stats(mock_database):
    """Test the get_all_usage_stats function."""
    mock_database.fetch_all.return_value = []
    await services.get_all_usage_stats()
    mock_database.fetch_all.assert_called_once()

@pytest.mark.asyncio
async def test_update_usage_stats_new_record(mock_database):
    """Test the update_usage_stats function for a new record."""
    mock_database.fetch_one.return_value = None
    await services.update_usage_stats("api_key", "model_name", 100)
    assert mock_database.execute.call_count == 1

@pytest.mark.asyncio
async def test_update_usage_stats_existing_record(mock_database):
    """Test the update_usage_stats function for an existing record."""
    mock_database.fetch_one.return_value = {"id": 1}
    await services.update_usage_stats("api_key", "model_name", 100)
    assert mock_database.execute.call_count == 1
