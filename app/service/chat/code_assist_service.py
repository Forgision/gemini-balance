# app/service/chat/code_assist_service.py

# app/service/chat/code_assist_service.py

import os
import json
import datetime
from typing import List, Optional
import google.oauth2.credentials
import google_auth_oauthlib.flow
from google.auth.transport.httpx import Request as AuthRequest
from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
import httpx
from pydantic import BaseModel

class StoredCredentials(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expiry_date: str

class OAuthWebConfig(BaseModel):
    client_id: str
    client_secret: str
    redirect_uris: List[str]
    auth_uri: str
    token_uri: str

class OAuthClientConfig(BaseModel):
    web: OAuthWebConfig

class CodeAssistService:
    OAUTH_SCOPE = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]
    CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"
    CODE_ASSIST_API_VERSION = "v1internal"

    def __init__(
        self,
        client_secrets_file="client_secret.json",
        credentials_file="credentials.json",
    ):
        with open(client_secrets_file, "r") as f:
            config_data = json.load(f)

        client_config = OAuthClientConfig(**config_data)
        self.client_config = client_config.model_dump()

        self.flow = google_auth_oauthlib.flow.Flow.from_client_config(
            self.client_config, scopes=self.OAUTH_SCOPE
        )
        self.flow.redirect_uri = client_config.web.redirect_uris[0]
        self.credentials_file = credentials_file
        self.credentials = self._load_credentials()
        self.client = httpx.AsyncClient()

    def _load_credentials(self):
        if not os.path.exists(self.credentials_file):
            return None
        try:
            with open(self.credentials_file, "r") as f:
                cred_data = json.load(f)
            stored_creds = StoredCredentials(**cred_data)
            creds = google.oauth2.credentials.Credentials(
                token=stored_creds.access_token,
                refresh_token=stored_creds.refresh_token,
                token_uri=self.client_config["web"]["token_uri"],
                client_id=self.client_config["web"]["client_id"],
                client_secret=self.client_config["web"]["client_secret"],
                scopes=self.OAUTH_SCOPE,
            )
            creds.expiry = datetime.datetime.fromisoformat(stored_creds.expiry_date)
            return creds
        except (IOError, json.JSONDecodeError, BaseModel.ValidationError) as e:
            # Handle corrupted or invalid credentials file
            print(f"Error loading credentials: {e}")
            return None

    def _save_credentials(self):
        if not self.credentials:
            return

        stored_creds = StoredCredentials(
            access_token=self.credentials.token,
            refresh_token=self.credentials.refresh_token,
            expiry_date=self.credentials.expiry.isoformat(),
        )

        with open(self.credentials_file, "w") as f:
            f.write(stored_creds.model_dump_json(indent=2))

    def get_authorization_url(self):
        authorization_url, state = self.flow.authorization_url(
            access_type="offline", include_granted_scopes="true"
        )
        return authorization_url

    def fetch_token(self, code):
        self.flow.fetch_token(code=code)
        self.credentials = self.flow.credentials
        self._save_credentials()

    def _get_method_url(self, method: str) -> str:
        return f"{self.CODE_ASSIST_ENDPOINT}/{self.CODE_ASSIST_API_VERSION}:{method}"

    async def _request_post(self, method: str, payload: dict):
        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                try:
                    self.credentials.refresh(AuthRequest(self.client))
                    self._save_credentials()
                except RefreshError as e:
                    # If refresh fails, re-authentication is needed
                    raise Exception("Token refresh failed. Please re-authenticate.") from e
            else:
                raise Exception("User not authenticated or credentials expired.")

        headers = {
            "Authorization": f"Bearer {self.credentials.token}",
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
