import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta
import json
from app.service.files.files_service import FilesService, _upload_sessions
from app.domain.file_models import FileMetadata, ListFilesResponse
from app.database.models import FileState

@pytest.fixture
def mock_async_client():
    """Fixture to mock httpx.AsyncClient."""
    mock_client_context = MagicMock()
    mock_instance = MagicMock()
    mock_instance.__aenter__.return_value = mock_client_context
    return mock_instance, mock_client_context

@pytest.fixture
def files_service():
    """Fixture to provide a FilesService instance."""
    _upload_sessions.clear()
    return FilesService()

@pytest.mark.asyncio
async def test_initialize_upload(files_service, mock_async_client):
    """Test the initialize_upload method."""
    mock_client_instance, mock_client_context = mock_async_client
    mock_post_response = MagicMock(
        status_code=200,
        headers={"x-goog-upload-url": "http://upload.url?upload_id=123"},
    )
    mock_client_context.post = AsyncMock(return_value=mock_post_response)

    mock_key_manager = MagicMock()
    mock_key_manager.get_next_key = AsyncMock(return_value="test_api_key")
    files_service.key_manager = mock_key_manager

    headers = {"x-goog-upload-protocol": "resumable"}
    body = json.dumps({"displayName": "test_file"}).encode()

    with patch("app.service.files.files_service.AsyncClient", return_value=mock_client_instance):
        response_body, response_headers = await files_service.initialize_upload(
            headers, body, "test_user", "http://localhost"
        )

    assert "X-Goog-Upload-URL" in response_headers
    assert "123" in _upload_sessions
    assert _upload_sessions["123"]["api_key"] == "test_api_key"

@pytest.mark.asyncio
async def test_get_file(files_service, mock_async_client):
    """Test the get_file method."""
    mock_client_instance, mock_client_context = mock_async_client
    now_iso = datetime.now(timezone.utc).isoformat()
    mock_get_response = MagicMock(
        status_code=200,
        json=MagicMock(return_value={
            "name": "files/test-file",
            "state": "ACTIVE",
            "mimeType": "text/plain",
            "sizeBytes": "1024",
            "createTime": now_iso,
            "updateTime": now_iso,
            "expirationTime": now_iso,
            "uri": "http://file.uri",
        })
    )
    mock_client_context.get = AsyncMock(return_value=mock_get_response)

    mock_file_record = {
        "name": "files/test-file",
        "api_key": "test_api_key",
        "expiration_time": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        "state": FileState.ACTIVE
    }

    with patch("app.service.files.files_service.db_services.get_file_record_by_name", AsyncMock(return_value=mock_file_record)), \
         patch("app.service.files.files_service.AsyncClient", return_value=mock_client_instance), \
         patch("app.service.files.files_service.db_services.update_file_record_state", new_callable=AsyncMock):

        file_metadata = await files_service.get_file("files/test-file", "test_user")

    assert isinstance(file_metadata, FileMetadata)
    assert file_metadata.name == "files/test-file"

@pytest.mark.asyncio
async def test_list_files(files_service):
    """Test the list_files method."""
    now = datetime.now(timezone.utc)
    mock_files = [
        {
            "name": "files/test-1", "display_name": "Test 1", "mime_type": "text/plain",
            "size_bytes": 100, "create_time": now, "update_time": now,
            "expiration_time": now + timedelta(days=1), "uri": "http://uri/1", "state": FileState.ACTIVE
        },
    ]

    with patch("app.service.files.files_service.db_services.list_file_records", AsyncMock(return_value=(mock_files, None))):
        response = await files_service.list_files()

    assert isinstance(response, ListFilesResponse)
    assert len(response.files) == 1
    assert response.files[0].name == "files/test-1"

@pytest.mark.asyncio
async def test_delete_file(files_service, mock_async_client):
    """Test the delete_file method."""
    mock_client_instance, mock_client_context = mock_async_client
    mock_delete_response = MagicMock(status_code=204)
    mock_client_context.delete = AsyncMock(return_value=mock_delete_response)

    mock_file_record = {"name": "files/test-delete", "api_key": "test_api_key"}

    with patch("app.service.files.files_service.db_services.get_file_record_by_name", AsyncMock(return_value=mock_file_record)), \
         patch("app.service.files.files_service.AsyncClient", return_value=mock_client_instance), \
         patch("app.service.files.files_service.db_services.delete_file_record", new_callable=AsyncMock) as mock_delete_db:

        success = await files_service.delete_file("files/test-delete", "test_user")

    assert success is True
    mock_delete_db.assert_called_once_with("files/test-delete")

@pytest.mark.asyncio
async def test_cleanup_expired_files(files_service, mock_async_client):
    """Test the cleanup_expired_files method."""
    mock_client_instance, mock_client_context = mock_async_client
    mock_client_context.delete = AsyncMock(return_value=MagicMock(status_code=204))

    mock_expired_files = [{"name": "files/expired-1", "api_key": "key-1"}]

    with patch("app.service.files.files_service.db_services.delete_expired_file_records", AsyncMock(return_value=mock_expired_files)), \
         patch("app.service.files.files_service.AsyncClient", return_value=mock_client_instance):

        count = await files_service.cleanup_expired_files()

    assert count == 1
    mock_client_context.delete.assert_called_once_with(
        "https://generativelanguage.googleapis.com/v1beta/files/expired-1", params={"key": "key-1"}
    )
