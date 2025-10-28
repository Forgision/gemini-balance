# app/service/chat/gemini_cli.py

import os
import json
import asyncio
import datetime
from typing import List, Optional
import requests
import google.oauth2.credentials
import google_auth_oauthlib.flow
from google.auth.transport.requests import Request as AuthRequest
from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
import httpx
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()


class GeminiCLIAuthorization(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expiry_date: str

class GeminiCLICredentialsWeb(BaseModel):
    client_id: str
    client_secret: str
    redirect_uris: List[str]
    auth_uri: str
    token_uri: str

class GeminiCLICredentials(BaseModel):
    web: GeminiCLICredentialsWeb

class GeminiCLIService:
    OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
    OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET")
    OAUTH_SCOPE = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]
    CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"
    CODE_ASSIST_API_VERSION = "v1internal"

    def __init__(self):
        self.authorization: Optional[google.oauth2.credentials.Credentials] = None
        self.credentials_file_path: Optional[str] = None
        self.client = httpx.AsyncClient()

    def load_credentials(self, json_path: str) -> GeminiCLICredentials:
        """Loads and validates the application credentials from a JSON file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Client secrets file not found at: {json_path}")
        try:
            with open(json_path, "r") as f:
                config_data = json.load(f)
            return GeminiCLICredentials(**config_data)
        except (IOError, json.JSONDecodeError, ValidationError) as e:
            print(f"Error loading client secrets file: {e}")
            raise

    def load_authorization(self, json_path: str) -> GeminiCLIAuthorization:
        """Loads, validates, and sets the user's authorization from a JSON file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Authorization file not found at: {json_path}")

        try:
            with open(json_path, "r") as f:
                cred_data = json.load(f)
            auth_model = GeminiCLIAuthorization(**cred_data)
        except (IOError, json.JSONDecodeError, ValidationError) as e:
            print(f"Error loading authorization file: {e}")
            raise

        creds = google.oauth2.credentials.Credentials(
            token=auth_model.access_token,
            refresh_token=auth_model.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=self.OAUTH_CLIENT_ID,
            client_secret=self.OAUTH_CLIENT_SECRET,
            scopes=self.OAUTH_SCOPE,
        )
        creds.expiry = datetime.datetime.fromisoformat(auth_model.expiry_date)

        if creds.expired and creds.refresh_token:
            creds.refresh(AuthRequest(requests.Session()))
            self._save_authorization(creds, json_path)

        self.authorization = creds
        self.credentials_file_path = json_path
        return self._get_authorization_model()

    def _get_authorization_model(self) -> GeminiCLIAuthorization:
        """Creates a GeminiCLIAuthorization model from the active credentials."""
        if not self.authorization:
            raise Exception("No active authorization.")
        if not self.authorization.token or not self.authorization.expiry:
            raise ValueError(
                "Cannot create authorization model from incomplete credentials."
            )
        return GeminiCLIAuthorization(
            access_token=self.authorization.token,
            refresh_token=self.authorization.refresh_token,
            token_type="Bearer",
            expiry_date=self.authorization.expiry.isoformat(),
        )

    def _save_authorization(self, credentials: google.oauth2.credentials.Credentials, json_path: str):
        """Saves the active credentials to a JSON file."""
        if not credentials or not credentials.token or not credentials.expiry:
            raise ValueError("Cannot save incomplete credentials.")
        auth_model = GeminiCLIAuthorization(
            access_token=credentials.token,
            refresh_token=credentials.refresh_token,
            token_type="Bearer",
            expiry_date=credentials.expiry.isoformat(),
        )
        with open(json_path, "w") as f:
            f.write(auth_model.model_dump_json(indent=2))

    def oauth(
        self,
        client_creds_path: Optional[str] = None,
        auth_file_path: str = "credentials.json",
    ) -> bool:
        """Handles the interactive, browser-based login flow."""
        client_config = self.load_credentials(client_creds_path).model_dump() if client_creds_path else {
            "web": {
                "client_id": self.OAUTH_CLIENT_ID,
                "client_secret": self.OAUTH_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost:8080/oauth2callback"],
            }
        }

        flow = google_auth_oauthlib.flow.Flow.from_client_config(
            client_config, scopes=self.OAUTH_SCOPE
        )
        flow.redirect_uri = client_config["web"]["redirect_uris"][0]
        
        auth_url, _ = flow.authorization_url(prompt="select_account")
        print("Please go to this URL: %s" % auth_url)

        code = input("Enter the authorization code: ")
        flow.fetch_token(code=code)

        self.authorization = flow.credentials
        if not self.authorization:
            raise Exception("OAuth flow failed to produce credentials.")
        self._save_authorization(self.authorization, auth_file_path)
        return True

    def _get_method_url(self, method: str) -> str:
        return f"{self.CODE_ASSIST_ENDPOINT}/{self.CODE_ASSIST_API_VERSION}:{method}"

    async def _request_post(self, method: str, payload: dict):
        if not self.authorization:
            raise Exception(
                "User not authenticated. Please call 'oauth' or 'load_authorization' first."
            )


        if not self.authorization.valid:
            if self.authorization.expired and self.authorization.refresh_token:
                import requests
                try:
                    await asyncio.to_thread(self.authorization.refresh, AuthRequest(requests.Session()))
                    if self.credentials_file_path:
                        self._save_authorization(self.authorization, self.credentials_file_path)
                except RefreshError as e:
                    raise Exception("Token refresh failed. Please re-authenticate.") from e
            else:
                raise Exception("Credentials expired and no refresh token available.")

        headers = {
            "Authorization": f"Bearer {self.authorization.token}",
            "Content-Type": "application/json",
        }
        url = self._get_method_url(method)
        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    async def generate_content(self, payload: dict):
        return await self._request_post("generateContent", payload)

    async def count_tokens(self, payload: dict):
        return await self._request_post("countTokens", payload)
