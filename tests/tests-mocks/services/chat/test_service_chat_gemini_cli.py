# tests/services/chat/test_service_chat_gemini_cli.py

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, mock_open, AsyncMock

import pytest
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

from app.service.chat.gemini_cli import (
    GeminiCLIAuthorization,
    GeminiCLICredentials,
    GeminiCLIService,
)


@pytest.fixture
def service():
    """Fixture to create a GeminiCLIService instance."""
    return GeminiCLIService()


@pytest.fixture
def mock_credentials_data():
    """Fixture for mock credentials data."""
    return {
        "web": {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "redirect_uris": ["http://localhost:8080/oauth2callback"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }


@pytest.fixture
def mock_auth_data():
    """Fixture for mock authorization data."""
    return {
        "access_token": "test_access_token",
        "refresh_token": "test_refresh_token",
        "token_type": "Bearer",
        "expiry_date": (datetime.now() + timedelta(hours=1)).isoformat(),
    }
    
    
@pytest.fixture
def mock_auth_data_expired(mock_auth_data):
    """Fixture for mock authorization data."""
    expired_auth_data = mock_auth_data.copy()
    now = datetime.now(timezone.utc)
    expiry_date = now - timedelta(hours=1)  
    expired_auth_data["expiry_date"] = expiry_date.strftime("%Y-%m-%dT%H:%M:%S.%f") # Make sure it's an offset-naive datetime
    return expired_auth_data


@pytest.fixture
def dumy_cred_path(mocker):
    exists = mocker.patch("os.path.exists")
    exists.return_value = True
    yield "dummy_path.json"
    exists.reset_mock()


def test_load_credentials_success(service, mock_credentials_data):
    """Test successful loading of credentials."""
    m = mock_open(read_data=json.dumps(mock_credentials_data))
    with patch("builtins.open", m), patch("os.path.exists", return_value=True):
        creds = service.load_credentials("dummy_path.json")
        assert isinstance(creds, GeminiCLICredentials)
        assert creds.web.client_id == "test_client_id"


def test_load_credentials_file_not_found(service):
    """Test FileNotFoundError when loading credentials."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            service.load_credentials("non_existent.json")


def test_load_credentials_invalid_json(service):
    """Test handling of invalid JSON when loading credentials."""
    m = mock_open(read_data="invalid json")
    with patch("builtins.open", m), patch("os.path.exists", return_value=True):
        with pytest.raises(json.JSONDecodeError):
            service.load_credentials("dummy_path.json")


@patch("google.oauth2.credentials.Credentials.refresh", return_value=None)
def test_load_authorization_success(mock_refresh, service, mock_auth_data, mock_credentials_data):
    """Test successful loading of authorization."""
    service.OAUTH_CLIENT_ID = mock_credentials_data["web"]["client_id"]
    service.OAUTH_CLIENT_SECRET = mock_credentials_data["web"]["client_secret"]
    m = mock_open(read_data=json.dumps(mock_auth_data))
    with patch("builtins.open", m), patch("os.path.exists", return_value=True):
        with patch.object(service, "_save_authorization") as mock_save:
            auth = service.load_authorization("dummy_auth.json")
            assert isinstance(auth, GeminiCLIAuthorization)
            assert service.authorization is not None
            assert service.authorization.token == "test_access_token"


def test_load_authorization_file_not_found(service):
    """Test FileNotFoundError when loading authorization."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            service.load_authorization("non_existent.json")


@patch("google.oauth2.credentials.Credentials.refresh", return_value=None)
def test_load_authorization_refresh(mock_refresh, service, mock_auth_data_expired, mock_credentials_data):
    """Test token refresh during authorization loading."""
    service.OAUTH_CLIENT_ID = mock_credentials_data["web"]["client_id"]
    service.OAUTH_CLIENT_SECRET = mock_credentials_data["web"]["client_secret"]
    m = mock_open(read_data=json.dumps(mock_auth_data_expired))

    with (
        patch("builtins.open", m),
        patch("os.path.exists", return_value=True),
        
    ):
        with patch.object(service, "_save_authorization") as mock_save:
            service.load_authorization("dummy_auth.json")
            mock_refresh.assert_called_once()
            mock_save.assert_called_once()


@patch("google_auth_oauthlib.flow.Flow.from_client_config")
@patch("builtins.input", return_value="test_code")
def test_oauth_flow(mock_input, mock_flow_from_config, service, mock_credentials_data):
    """Test the full OAuth flow."""
    mock_flow = MagicMock()
    mock_flow.authorization_url.return_value = ("http://auth.url", "state")
    mock_flow_from_config.return_value = mock_flow

    service.OAUTH_CLIENT_ID = mock_credentials_data["web"]["client_id"]
    service.OAUTH_CLIENT_SECRET = mock_credentials_data["web"]["client_secret"]

    with patch.object(service, "_save_authorization") as mock_save:
        result = service.oauth()
        assert result is True
        mock_flow.fetch_token.assert_called_with(code="test_code")
        mock_save.assert_called_once()
        assert service.authorization is not None


@pytest.mark.asyncio
async def test_request_post_success(service):
    """Test a successful POST request."""
    service.authorization = MagicMock(spec=Credentials)
    service.authorization.valid = True
    service.authorization.token = "test_token"

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"data": "success"}

    with patch.object(service.client, "post", return_value=mock_response) as mock_post:
        response = await service._request_post("testMethod", {"payload": "data"})
        mock_post.assert_called_once()
        assert response == {"data": "success"}


@pytest.mark.asyncio
async def test_request_post_unauthenticated(service):
    """Test request when not authenticated."""
    with pytest.raises(Exception, match="User not authenticated"):
        await service._request_post("testMethod", {})


@pytest.mark.asyncio
@patch("asyncio.to_thread")
async def test_request_post_token_refresh_failure(mock_to_thread, service):
    """Test request with a token refresh failure."""
    service.authorization = MagicMock(spec=Credentials)
    service.authorization.valid = False
    service.authorization.expired = True
    service.authorization.refresh_token = "refresh_token"
    mock_to_thread.side_effect = RefreshError("Refresh failed")

    with pytest.raises(Exception, match="Token refresh failed"):
        await service._request_post("testMethod", {})


@pytest.mark.asyncio
async def test_generate_content(service):
    """Test the generate_content method."""
    with patch.object(service, "_request_post", new_callable=AsyncMock) as mock_request:
        payload = {"contents": "Hello"}
        await service.generate_content(payload)
        mock_request.assert_called_once_with("generateContent", payload)


@pytest.mark.asyncio
async def test_count_tokens(service):
    """Test the count_tokens method."""
    with patch.object(service, "_request_post", new_callable=AsyncMock) as mock_request:
        payload = {"contents": "Hello"}
        await service.count_tokens(payload)
        mock_request.assert_called_once_with("countTokens", payload)
