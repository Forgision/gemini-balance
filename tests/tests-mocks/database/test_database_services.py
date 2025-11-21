import datetime
import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import services
from app.database.models import FileState


@pytest.fixture
def mock_session():
    """Fixture for mocking AsyncSession."""
    session = AsyncMock(spec=AsyncSession)
    # Mock the execute method and its return value
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_result.scalar_one_or_none.return_value = None
    mock_result.scalar_one.return_value = 0
    mock_result.first.return_value = None
    mock_result.fetchall.return_value = []
    session.execute.return_value = mock_result
    session.commit = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_get_all_settings(mock_session):
    """Test the get_all_settings function."""
    mock_session.execute.return_value.scalars.return_value.all.return_value = []
    result = await services.get_all_settings(mock_session)
    assert result == []
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_setting(mock_session):
    """Test the get_setting function."""
    mock_session.execute.return_value.scalar_one_or_none.return_value = None
    result = await services.get_setting(mock_session, "some_key")
    assert result is None
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_update_setting(mock_session):
    """Test the update_setting function."""
    from unittest.mock import patch

    with patch(
        "app.database.services.get_setting", new_callable=AsyncMock
    ) as mock_get_setting:
        mock_get_setting.return_value = {
            "key": "some_key",
            "value": "old_value",
            "description": "",
        }
        await services.update_setting(mock_session, "some_key", "new_value")
        mock_session.execute.assert_called()
        mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_add_error_log(mock_session):
    """Test the add_error_log function."""
    await services.add_error_log(
        mock_session, "gemini_key", "model_name", "error_type", "error_log"
    )
    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_get_error_logs(mock_session):
    """Test the get_error_logs function."""
    mock_session.execute.return_value.fetchall.return_value = []
    result = await services.get_error_logs(mock_session)
    assert result == []
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_delete_error_logs_by_ids(mock_session):
    """Test the delete_error_logs_by_ids function."""
    # delete_error_logs_by_ids does not commit (caller handles commit)
    mock_session.execute = AsyncMock(return_value=MagicMock())
    result = await services.delete_error_logs_by_ids(mock_session, [1, 2, 3])
    mock_session.execute.assert_called_once()
    assert result == 3  # Should return the number of IDs
    # Note: This function does NOT call commit


@pytest.mark.asyncio
async def test_create_file_record(mock_session):
    """Test the create_file_record function."""
    from unittest.mock import patch

    with patch(
        "app.database.services.get_file_record_by_name", new_callable=AsyncMock
    ) as mock_get_file_record_by_name:
        mock_get_file_record_by_name.return_value = {"name": "files/file_id"}
        await services.create_file_record(
            mock_session,
            "files/file_id",
            "mime/type",
            123,
            "api_key",
            "uri",
            datetime.datetime(2024, 1, 1),
            datetime.datetime(2025, 1, 1),
            datetime.datetime(2025, 1, 1),
        )
        mock_session.execute.assert_called()
        mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_get_file_record_by_name(mock_session):
    """Test the get_file_record_by_name function."""
    mock_session.execute.return_value.scalar_one_or_none.return_value = None
    result = await services.get_file_record_by_name(mock_session, "files/file_id")
    assert result is None
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_update_file_record_state(mock_session):
    """Test the update_file_record_state function."""
    from unittest.mock import patch

    with patch(
        "app.database.services.get_file_record_by_name", new_callable=AsyncMock
    ) as mock_get:
        mock_get.return_value = {"name": "files/file_id"}
        await services.update_file_record_state(
            mock_session, "files/file_id", FileState.ACTIVE
        )
        mock_session.execute.assert_called()
        mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_list_file_records(mock_session):
    """Test the list_file_records function."""
    mock_session.execute.return_value.scalars.return_value.all.return_value = []
    result, token = await services.list_file_records(mock_session)
    assert result == []
    assert token is None
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_delete_file_record(mock_session):
    """Test the delete_file_record function."""
    await services.delete_file_record(mock_session, "files/file_id")
    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_expired_file_records(mock_session):
    """Test the delete_expired_file_records function."""
    # When no expired records, function returns early after select
    mock_result_select = MagicMock()
    mock_result_select.scalars.return_value.all.return_value = []

    mock_session.execute = AsyncMock(return_value=mock_result_select)
    mock_session.commit = AsyncMock()

    result = await services.delete_expired_file_records(mock_session)
    assert result == []
    assert mock_session.execute.call_count == 1  # Only select is called when no records
    mock_session.commit.assert_not_called()  # Commit not called when no records to delete


@pytest.mark.asyncio
async def test_delete_expired_file_records_with_records(mock_session):
    """Test the delete_expired_file_records function when records exist."""
    # When expired records exist, function calls execute twice (select then delete)
    mock_record = MagicMock()
    mock_record.__dict__ = {"name": "files/file_id", "api_key": "api_key"}

    mock_result_select = MagicMock()
    mock_result_select.scalars.return_value.all.return_value = [mock_record]

    mock_result_delete = MagicMock()

    # Set up execute to return different results for select and delete
    mock_session.execute = AsyncMock(
        side_effect=[mock_result_select, mock_result_delete]
    )
    mock_session.commit = AsyncMock()

    result = await services.delete_expired_file_records(mock_session)
    assert len(result) == 1
    assert mock_session.execute.call_count == 2  # Called twice: select and delete
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_get_usage_stats_by_key_and_model(mock_session):
    """Test the get_usage_stats_by_key_and_model function."""
    mock_session.execute.return_value.scalar_one_or_none.return_value = None
    result = await services.get_usage_stats_by_key_and_model(
        mock_session, "api_key", "model_name"
    )
    assert result is None
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_all_usage_stats(mock_session):
    """Test the get_all_usage_stats function."""
