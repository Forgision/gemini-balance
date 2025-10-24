"""
Static resource versioning tool
Used to add version parameters to CSS and JS files to avoid browser caching issues
"""

import hashlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict

from app.utils.helpers import get_current_version


class StaticVersionManager:
    """Static resource version manager"""

    def __init__(self, static_dir: str = "app/static"):
        self.static_dir = Path(static_dir)
        self._version_cache: Dict[str, str] = {}
        self._use_file_hash = True  # Whether to use file hash as version number

    def get_version_for_file(self, file_path: str) -> str:
        """
        Get the version number of a file

        Args:
            file_path: File path relative to the static directory, e.g., 'css/fonts.css'

        Returns:
            Version number string
        """
        if self._use_file_hash:
            return self._get_file_hash_version(file_path)
        else:
            return self._get_app_version()

    def _get_file_hash_version(self, file_path: str) -> str:
        """Generate a hash version number based on file content"""
        # If already cached, return directly
        if file_path in self._version_cache:
            return self._version_cache[file_path]

        full_path = self.static_dir / file_path

        if not full_path.exists():
            # If the file does not exist, use the application version number as a fallback
            version = self._get_app_version()
        else:
            try:
                # Read file content and calculate MD5 hash
                with open(full_path, "rb") as f:
                    content = f.read()
                    hash_object = hashlib.md5(content)
                    version = hash_object.hexdigest()[:8]  # Take the first 8 characters
            except Exception:
                # If reading fails, use the application version number as a fallback
                version = self._get_app_version()

        # Cache the result
        self._version_cache[file_path] = version
        return version

    def _get_app_version(self) -> str:
        """Get the application version number"""
        try:
            return get_current_version().replace(".", "")
        except Exception:
            # If getting the version fails, use a timestamp
            return str(int(time.time()))

    def get_versioned_url(self, file_path: str) -> str:
        """
        Get a URL with a version parameter

        Args:
            file_path: File path relative to the static directory

        Returns:
            URL with version parameter
        """
        version = self.get_version_for_file(file_path)
        return f"/static/{file_path}?v={version}"

    def clear_cache(self):
        """Clear the version cache"""
        self._version_cache.clear()


# Global instance
_static_version_manager = StaticVersionManager()


def get_static_url(file_path: str) -> str:
    """
    Get the versioned URL for a static resource

    Args:
        file_path: File path relative to the static directory

    Returns:
        Complete URL with version parameter

    Example:
        get_static_url('css/fonts.css') -> '/static/css/fonts.css?v=a1b2c3d4'
        get_static_url('js/config_editor.js') -> '/static/js/config_editor.js?v=e5f6g7h8'
    """
    return _static_version_manager.get_versioned_url(file_path)


def clear_static_cache():
    """Clear the static resource version cache"""
    _static_version_manager.clear_cache()


@lru_cache(maxsize=128)
def get_cached_static_url(file_path: str) -> str:
    """
    Get the cached URL for a static resource (for development environment)

    Args:
        file_path: File path relative to the static directory

    Returns:
        Complete URL with version parameter
    """
    return get_static_url(file_path)
