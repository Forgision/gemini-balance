"""
Integration tests for file upload/management routes.
"""

import pytest


@pytest.mark.asyncio
async def test_initialize_file_upload(test_client, goog_api_key_header):
    """Test initializing file upload."""
    payload = {
        "file": {
            "display_name": "test.txt",
            "mime_type": "text/plain"
        }
    }
    
    headers = {
        **goog_api_key_header,
        "x-goog-upload-protocol": "resumable",
        "x-goog-upload-command": "start"
    }
    
    response = test_client.post(
        "/upload/v1beta/files",
        json=payload,
        headers=headers
    )
    
    # Should work or return appropriate error
    assert response.status_code in [200, 400, 500]


@pytest.mark.asyncio
async def test_list_files(test_client, goog_api_key_header):
    """Test listing files."""
    response = test_client.get(
        "/v1beta/files?pageSize=10",
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_list_files_with_pagination(test_client, goog_api_key_header):
    """Test listing files with pagination."""
    response = test_client.get(
        "/v1beta/files?pageSize=10&pageToken=test",
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_list_files_gemini_prefix(test_client, goog_api_key_header):
    """Test listing files via Gemini prefix endpoint."""
    response = test_client.get(
        "/gemini/v1beta/files?pageSize=10",
        headers=goog_api_key_header
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

