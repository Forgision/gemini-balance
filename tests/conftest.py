from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

from dotenv import load_dotenv
import pytest

# Add the project root to the Python path to ensure app modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load test environment variables before importing any application modules
# This is crucial to ensure the app is configured for testing
load_dotenv(dotenv_path=PROJECT_ROOT / ".env.test", override=True)


@pytest.fixture(autouse=True)
def mock_check_for_updates():
    """
    Mock the check_for_updates function to prevent network calls to GitHub during tests.
    This avoids API rate limit errors and speeds up test execution.
    """
    with patch(
        "app.service.update.update_service.check_for_updates", new_callable=AsyncMock
    ) as mock:
        mock.return_value = (False, None, None)
        yield mock
