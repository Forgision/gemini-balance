import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from app.utils.helpers import (
    extract_mime_type_and_data,
    convert_image_to_base64,
    parse_prompt_parameters,
    extract_image_urls_from_markdown,
    is_valid_api_key,
    get_current_version,
    is_image_upload_configured,
)
from app.config.config import Settings

def test_extract_mime_type_and_data():
    """Test the extract_mime_type_and_data function."""
    mime_type, data = extract_mime_type_and_data("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
    assert mime_type == "image/png"
    assert data == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

@patch("requests.get")
def test_convert_image_to_base64(mock_get):
    """Test the convert_image_to_base64 function."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"image_data"
    mock_get.return_value = mock_response
    result = convert_image_to_base64("http://example.com/image.jpg")
    assert result == "aW1hZ2VfZGF0YQ=="

def test_parse_prompt_parameters():
    """Test the parse_prompt_parameters function."""
    prompt, n, aspect_ratio = parse_prompt_parameters("a cat {n:2} {ratio:16:9}")
    assert prompt == "a cat"
    assert n == 2
    assert aspect_ratio == "16:9"

def test_extract_image_urls_from_markdown():
    """Test the extract_image_urls_from_markdown function."""
    urls = extract_image_urls_from_markdown("![image](http://example.com/image.jpg)")
    assert urls == ["http://example.com/image.jpg"]

def test_is_valid_api_key():
    """Test the is_valid_api_key function."""
    assert is_valid_api_key("AIza" + "a" * 26) is True
    assert is_valid_api_key("sk-" + "a" * 28) is True
    assert is_valid_api_key("invalid_key") is False

def test_get_current_version(tmp_path):
    """Test the get_current_version function."""
    version_file = tmp_path / "VERSION"
    version_file.write_text("1.2.3")
    assert get_current_version(version_file) == "1.2.3"

def test_is_image_upload_configured():
    """Test the is_image_upload_configured function."""
    settings = Settings()
    settings.UPLOAD_PROVIDER = "smms"
    settings.SMMS_SECRET_TOKEN = "token"
    assert is_image_upload_configured(settings) is True
