# app/service/chat/code_assist_service.py

# app/service/chat/code_assist_service.py

import os
from typing import List
import google.oauth2.credentials
import google_auth_oauthlib.flow
from googleapiclient.discovery import build
import httpx
from pydantic import BaseModel, Field

class OAuthWebConfig(BaseModel):
    client_id: str
    client_secret: str
    auth_uri: str = "https://accounts.google.com/o/oauth2/auth"
    token_uri: str = "https://oauth2.googleapis.com/token"
    redirect_uris: List[str]

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

    def __init__(self):
        redirect_uri = os.environ.get(
            "OAUTH_REDIRECT_URI", "http://localhost:8080/oauth2callback"
        )
        web_config = OAuthWebConfig(
            client_id=os.environ.get("OAUTH_CLIENT_ID"),
            client_secret=os.environ.get("OAUTH_CLIENT_SECRET"),
            redirect_uris=[redirect_uri],
        )
        client_config = OAuthClientConfig(web=web_config)

        self.flow = google_auth_oauthlib.flow.Flow.from_client_config(
            client_config.model_dump(),
            scopes=self.OAUTH_SCOPE,
        )
        self.flow.redirect_uri = redirect_uri
        self.credentials = None
        self.client = httpx.AsyncClient()

    def get_authorization_url(self):
        authorization_url, state = self.flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
        )
        return authorization_url

    def fetch_token(self, code):
        self.flow.fetch_token(code=code)
        self.credentials = self.flow.credentials

    def _get_method_url(self, method: str) -> str:
        return f"{self.CODE_ASSIST_ENDPOINT}/{self.CODE_ASSIST_API_VERSION}:{method}"

    async def _request_post(self, method: str, payload: dict):
        if not self.credentials or not self.credentials.valid:
            # Here you would typically refresh the token or re-authenticate
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
