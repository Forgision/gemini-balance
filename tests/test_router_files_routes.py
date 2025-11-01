from unittest.mock import patch, AsyncMock
import pytest
from fastapi import HTTPException
from app.dependencies import get_files_service
from app.domain.file_models import FileMetadata, ListFilesResponse

@pytest.fixture(autouse=True)
def override_dependencies(client):
    # This fixture ensures that for every test in this file,
    # the get_files_service dependency is mocked.
    # It's cleared after each test to prevent side effects.
    client.app.dependency_overrides[get_files_service] = AsyncMock()
    yield
    client.app.dependency_overrides.clear()


@patch("app.core.security.settings.ALLOWED_TOKENS", ["test_auth_token"])
async def test_list_files_success(client, mock_key_manager):
    """Test successful listing of files."""
    mock_files_service = AsyncMock()
    mock_files_service.list_files.return_value = ListFilesResponse(files=[], nextPageToken=None)
    client.app.dependency_overrides[get_files_service] = lambda: mock_files_service

    response = client.get("/v1beta/files", headers={"x-goog-api-key": "test_auth_token"})
    assert response.status_code == 200
    assert response.json() == {"files": [], "nextPageToken": None}
    mock_files_service.list_files.assert_called_once()


def test_list_files_unauthorized(client):
    """Test unauthorized access to list_files."""
    response = client.get("/v1beta/files")
    assert response.status_code == 401


@patch("app.core.security.settings.ALLOWED_TOKENS", ["test_auth_token"])
async def test_get_file_success(client, mock_key_manager):
    """Test successful retrieval of a file."""
    mock_files_service = AsyncMock()
    mock_files_service.get_file.return_value = FileMetadata(
        name="files/test_file",
        displayName="test_file",
        mimeType="text/plain",
        sizeBytes="1234",
        createTime="2025-10-31T12:00:00Z",
        updateTime="2025-10-31T12:00:00Z",
        expirationTime="2025-11-30T12:00:00Z",
        uri="http://example.com/test_file",
        state="ACTIVE",
    )
    client.app.dependency_overrides[get_files_service] = lambda: mock_files_service

    response = client.get("/v1beta/files/test_file", headers={"x-goog-api-key": "test_auth_token"})
    assert response.status_code == 200
    assert response.json()["name"] == "files/test_file"
    mock_files_service.get_file.assert_called_once_with("files/test_file", "test_auth_token")


def test_get_file_unauthorized(client):
    """Test unauthorized access to get_file."""
    response = client.get("/v1beta/files/test_file")
    assert response.status_code == 401


@patch("app.core.security.settings.ALLOWED_TOKENS", ["test_auth_token"])
async def test_delete_file_success(client, mock_key_manager):
    """Test successful deletion of a file."""
    mock_files_service = AsyncMock()
    mock_files_service.delete_file.return_value = True
    client.app.dependency_overrides[get_files_service] = lambda: mock_files_service

    response = client.delete("/v1beta/files/test_file", headers={"x-goog-api-key": "test_auth_token"})
    assert response.status_code == 200
    assert response.json()["success"] is True
    mock_files_service.delete_file.assert_called_once_with("files/test_file", "test_auth_token")


def test_delete_file_unauthorized(client):
    """Test unauthorized access to delete_file."""
    response = client.delete("/v1beta/files/test_file")
    assert response.status_code == 401


@patch("app.core.security.settings.ALLOWED_TOKENS", ["test_auth_token"])
async def test_gemini_prefixed_routes(client, mock_key_manager):
    """Test that the Gemini-prefixed routes work correctly."""
    mock_files_service = AsyncMock()
    mock_files_service.list_files.return_value = ListFilesResponse(files=[], nextPageToken=None)
    client.app.dependency_overrides[get_files_service] = lambda: mock_files_service

    response = client.get("/gemini/v1beta/files", headers={"x-goog-api-key": "test_auth_token"})
    assert response.status_code == 200
    assert response.json() == {"files": [], "nextPageToken": None}


@patch("app.core.security.settings.ALLOWED_TOKENS", ["test_auth_token"])
async def test_upload_file_init_success(client, mock_key_manager):
    """Test successful file upload initialization."""
    mock_files_service = AsyncMock()
    mock_files_service.initialize_upload.return_value = ({}, {})
    client.app.dependency_overrides[get_files_service] = lambda: mock_files_service

    response = client.post("/upload/v1beta/files", json={}, headers={"x-goog-api-key": "test_auth_token"})
    assert response.status_code == 200


@patch("app.core.security.settings.ALLOWED_TOKENS", ["test_auth_token"])
@patch("app.router.files_routes.get_upload_handler")
async def test_handle_upload_success(mock_get_upload_handler, client, mock_key_manager):
    """Test successful handling of a file upload."""
    mock_files_service = AsyncMock()
    mock_files_service.get_upload_session.return_value = {
        "api_key": "test_api_key",
        "upload_url": "http://test_upload_url",
    }
    client.app.dependency_overrides[get_files_service] = lambda: mock_files_service

    mock_upload_handler = AsyncMock()
    mock_upload_handler.proxy_upload_request.return_value = {}
    mock_get_upload_handler.return_value = mock_upload_handler

    response = client.post("/upload/v1beta/files?upload_id=test_upload_id", headers={"x-goog-api-key": "test_auth_token"})
    assert response.status_code == 200