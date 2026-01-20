from unittest.mock import AsyncMock, patch
import pytest

@pytest.fixture(autouse=True)
def mock_check_for_updates():
    """
    Mock the check_for_updates function to prevent network calls to GitHub.
    """
    with patch(
        "app.service.update.update_service.check_for_updates", new_callable=AsyncMock
    ) as mock:
        mock.return_value = (False, None, None)
        yield mock
