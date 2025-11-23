import pytest
from unittest.mock import patch
from app.utils.static_version import StaticVersionManager, get_static_url, clear_static_cache

@pytest.fixture
def manager():
    """Fixture for StaticVersionManager."""
    return StaticVersionManager()

def test_static_version_manager_get_version_for_file(manager):
    """Test the StaticVersionManager.get_version_for_file method."""
    with patch("builtins.open") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"file_content"
        with patch("app.utils.static_version.Path.exists") as mock_exists:
            mock_exists.return_value = True
            version = manager.get_version_for_file("main.css")
            assert version is not None

def test_get_static_url():
    """Test the get_static_url function."""
    with patch("app.utils.static_version._static_version_manager.get_versioned_url") as mock_get_versioned_url:
        mock_get_versioned_url.return_value = "/static/main.css?v=123"
        url = get_static_url("main.css")
        assert url == "/static/main.css?v=123"

def test_clear_static_cache():
    """Test the clear_static_cache function."""
    with patch("app.utils.static_version._static_version_manager.clear_cache") as mock_clear_cache:
        clear_static_cache()
        mock_clear_cache.assert_called_once()
