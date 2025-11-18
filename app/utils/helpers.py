"""
General utility functions module
"""

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.config.config import Settings
from app.core.constants import DATA_URL_PATTERN, IMAGE_URL_PATTERN, VALID_IMAGE_RATIOS

helper_logger = logging.getLogger("app.utils")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VERSION_FILE_PATH = PROJECT_ROOT / "VERSION"


def extract_mime_type_and_data(base64_string: str) -> Tuple[Optional[str], str]:
    """
    Extracts MIME type and data from a base64 string.

    Args:
        base64_string: A base64 string that may contain MIME type information.

    Returns:
        tuple: (mime_type, encoded_data)
    """
    # Check if the string starts with the "data:" format.
    if base64_string.startswith("data:"):
        # Extract MIME type and data.
        pattern = DATA_URL_PATTERN
        match = re.match(pattern, base64_string)
        if match:
            mime_type = (
                "image/jpeg" if match.group(1) == "image/jpg" else match.group(1)
            )
            encoded_data = match.group(2)
            return mime_type, encoded_data

    # If it's not in the expected format, assume it's just the data part.
    return None, base64_string


def convert_image_to_base64(url: str) -> str:
    """
    Converts an image URL to base64 encoding.

    Args:
        url: The URL of the image.

    Returns:
        str: The base64-encoded image data.

    Raises:
        Exception: If fetching the image fails.
    """
    response = requests.get(url)
    if response.status_code == 200:
        # Convert the image content to base64.
        img_data = base64.b64encode(response.content).decode("utf-8")
        return img_data
    else:
        raise Exception(f"Failed to fetch image: {response.status_code}")


def format_json_response(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Formats a JSON response.

    Args:
        data: The data to format.
        indent: The number of spaces for indentation.

    Returns:
        str: The formatted JSON string.
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def parse_prompt_parameters(
    prompt: str, default_ratio: str = "1:1"
) -> Tuple[str, int, str]:
    """
    Parses parameters from a prompt.

    Supported formats:
    - {n:number} e.g.: {n:2} generates 2 images
    - {ratio:aspect_ratio} e.g.: {ratio:16:9} uses a 16:9 aspect ratio

    Args:
        prompt: The prompt text.
        default_ratio: The default aspect ratio.

    Returns:
        tuple: (cleaned_prompt_text, number_of_images, aspect_ratio)
    """
    # Default values
    n = 1
    aspect_ratio = default_ratio

    # Parse n parameter
    n_match = re.search(r"{n:(\d+)}", prompt)
    if n_match:
        n = int(n_match.group(1))
        if n < 1 or n > 4:
            raise ValueError(f"Invalid n value: {n}. Must be between 1 and 4.")
        prompt = prompt.replace(n_match.group(0), "").strip()

    # Parse ratio parameter
    ratio_match = re.search(r"{ratio:(\d+:\d+)}", prompt)
    if ratio_match:
        aspect_ratio = ratio_match.group(1)
        if aspect_ratio not in VALID_IMAGE_RATIOS:
            raise ValueError(
                f"Invalid ratio: {aspect_ratio}. Must be one of: {', '.join(VALID_IMAGE_RATIOS)}"
            )
        prompt = prompt.replace(ratio_match.group(0), "").strip()

    return prompt, n, aspect_ratio


def extract_image_urls_from_markdown(text: str) -> List[str]:
    """
    Extracts image URLs from Markdown text.

    Args:
        text: The Markdown text.

    Returns:
        List[str]: A list of image URLs.
    """
    pattern = IMAGE_URL_PATTERN
    matches = re.findall(pattern, text)
    return [match[1] for match in matches]


def is_valid_api_key(key: str) -> bool:
    """
    Checks if the API key format is valid.

    Args:
        key: The API key.

    Returns:
        bool: True if the key format is valid, otherwise False.
    """
    # Check Gemini API key format
    if key.startswith("AIza"):
        return len(key) >= 30

    # Check OpenAI API key format
    if key.startswith("sk-"):
        return len(key) >= 30

    return False


def redact_key_for_logging(key: Optional[str]) -> str:
    """
    Redacts API key for secure logging by showing only first and last 6 characters.

    Args:
        key: API key to redact

    Returns:
        str: Redacted key in format "first6...last6" or descriptive placeholder for edge cases
    """
    if not isinstance(key, str) or not key:
        return "[INVALID_KEY]"

    if len(key) <= 12:
        return "[SHORT_KEY]"
    else:
        return f"{key[:6]}...{key[-6:]}"


def get_current_version(
    version_file_path: Path = VERSION_FILE_PATH, default_version: str = "0.0.0"
) -> str:
    """Reads the current version from the specified version file."""
    try:
        with version_file_path.open("r", encoding="utf-8") as f:
            version = f.read().strip()
        if not version:
            helper_logger.warning(
                f"Version file ('{version_file_path}') is empty. Using default version '{default_version}'."
            )
            return default_version
        return version
    except FileNotFoundError:
        helper_logger.warning(
            f"Version file not found at '{version_file_path}'. Using default version '{default_version}'."
        )
        return default_version
    except IOError as e:
        helper_logger.error(
            f"Error reading version file ('{version_file_path}'): {e}. Using default version '{default_version}'.",
            exc_info=True,
        )
        return default_version


def is_image_upload_configured(settings: Settings) -> bool:
    """Return True only if a valid upload provider is selected and all required settings for that provider are present."""

    provider = (getattr(settings, "UPLOAD_PROVIDER", "") or "").strip().lower()
    if provider == "smms":
        return bool(getattr(settings, "SMMS_SECRET_TOKEN", ""))
    if provider == "picgo":
        return bool(getattr(settings, "PICGO_API_KEY", ""))
    if provider == "aliyun_oss":
        return all(
            [
                getattr(settings, "OSS_ACCESS_KEY", ""),
                getattr(settings, "OSS_ACCESS_KEY_SECRET", ""),
                getattr(settings, "OSS_BUCKET_NAME", ""),
                getattr(settings, "OSS_ENDPOINT", ""),
                getattr(settings, "OSS_REGION", ""),
            ]
        )
    if provider == "cloudflare_imgbed":
        return all(
            [
                getattr(settings, "CLOUDFLARE_IMGBED_URL", ""),
                getattr(settings, "CLOUDFLARE_IMGBED_AUTH_CODE", ""),
            ]
        )
    return False
