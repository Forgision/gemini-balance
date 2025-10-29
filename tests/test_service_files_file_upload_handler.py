import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request
from fastapi.responses import Response
from datetime import datetime, timezone

from app.service.files.file_upload_handler import FileUploadHandler

@pytest.fixture
def mock_async_client():
    """Fixture to mock httpx.AsyncClient."""
    mock_client_context = MagicMock()
    mock_instance = MagicMock()
    mock_instance.__aenter__.return_value = mock_client_context
    return mock_instance, mock_client_context

@pytest.mark.asyncio
async def test_handle_upload_chunk_partial(mock_async_client):
    """Test the FileUploadHandler.handle_upload_chunk method for a partial upload."""
    handler = FileUploadHandler()
    mock_client_instance, mock_client_context = mock_async_client

    mock_request = MagicMock(spec=Request)
    mock_request.headers = {
        "x-goog-upload-command": "upload",
        "x-goog-upload-offset": "0",
        "content-type": "application/octet-stream",
        "content-length": "1024",
    }
    mock_request.body = AsyncMock(return_value=b"test chunk")

    mock_post_response = MagicMock(status_code=308, headers={"x-goog-upload-status": "active"}, content=b"")
    mock_client_context.post = AsyncMock(return_value=mock_post_response)

    with patch("app.service.files.file_upload_handler.AsyncClient", return_value=mock_client_instance):
        response = await handler.handle_upload_chunk("http://upload.url", mock_request)

        mock_client_context.post.assert_called_once_with(
            "http://upload.url",
            headers={
                'X-Goog-Upload-Command': 'upload',
                'X-Goog-Upload-Offset': '0',
                'Content-Type': 'application/octet-stream',
                'Content-Length': '1024'
            },
            content=b"test chunk",
            timeout=300.0,
        )
        assert isinstance(response, Response)
        assert response.status_code == 308

@pytest.mark.asyncio
async def test_handle_upload_chunk_final(mock_async_client):
    """Test the FileUploadHandler.handle_upload_chunk method for a final upload."""
    handler = FileUploadHandler()
    mock_client_instance, mock_client_context = mock_async_client

    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"x-goog-upload-command": "upload, finalize"}
    mock_request.body = AsyncMock(return_value=b"last chunk")

    now_iso = datetime.now(timezone.utc).isoformat()
    mock_post_response = MagicMock(
        status_code=200,
        headers={"content-type": "application/json"},
        json=MagicMock(return_value={
            "file": {
                "name": "files/test-file-id",
                "state": "ACTIVE",
                "mimeType": "text/plain",
                "sizeBytes": "1536",
                "uri": "http://files.uri/test-file-id",
                "createTime": now_iso,
                "updateTime": now_iso,
                "expirationTime": now_iso,
            }
        })
    )
    mock_client_context.post = AsyncMock(return_value=mock_post_response)

    mock_files_service = MagicMock()
    mock_files_service.get_upload_session = AsyncMock(return_value={
        "api_key": "test_key",
        "user_token": "test_user",
        "mime_type": "text/plain",
        "size_bytes": "1536",
    })

    with patch("app.service.files.file_upload_handler.AsyncClient", return_value=mock_client_instance), \
         patch("app.service.files.file_upload_handler.db_services.create_file_record", new_callable=AsyncMock) as mock_create_record:

        await handler.handle_upload_chunk("http://upload.url", mock_request, files_service=mock_files_service)

        mock_files_service.get_upload_session.assert_called_once_with("http://upload.url")
        mock_create_record.assert_called_once()
        assert mock_create_record.call_args.kwargs["name"] == "files/test-file-id"

@pytest.mark.asyncio
async def test_proxy_upload_request_get():
    """Test proxying a GET request."""
    handler = FileUploadHandler()
    mock_request = MagicMock(spec=Request)
    mock_request.method = "GET"

    with patch.object(handler, '_get_upload_status', new_callable=AsyncMock) as mock_get_status:
        await handler.proxy_upload_request(mock_request, "http://upload.url")
        mock_get_status.assert_called_once_with("http://upload.url")

@pytest.mark.asyncio
async def test_proxy_upload_request_post():
    """Test proxying a POST request."""
    handler = FileUploadHandler()
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_files_service = MagicMock()

    with patch.object(handler, 'handle_upload_chunk', new_callable=AsyncMock) as mock_handle_chunk:
        await handler.proxy_upload_request(mock_request, "http://upload.url", files_service=mock_files_service)
        mock_handle_chunk.assert_called_once_with("http://upload.url", mock_request, mock_files_service)
