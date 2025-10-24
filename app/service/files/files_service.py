"""
File Management Service
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
from httpx import AsyncClient
import asyncio

from app.config.config import settings
from app.database import services as db_services
from app.database.models import FileState
from app.domain.file_models import FileMetadata, ListFilesResponse
from fastapi import HTTPException
from app.log.logger import get_files_logger
from app.utils.helpers import redact_key_for_logging
from app.service.client.api_client import GeminiApiClient
from app.service.key.key_manager import get_key_manager_instance

logger = get_files_logger()

# Global upload session storage
_upload_sessions: Dict[str, Dict[str, Any]] = {}
_upload_sessions_lock = asyncio.Lock()


class FilesService:
    """File management service class"""

    def __init__(self):
        self.api_client = GeminiApiClient(base_url=settings.BASE_URL)
        self.key_manager = None

    async def _get_key_manager(self):
        """Get KeyManager instance"""
        if not self.key_manager:
            self.key_manager = await get_key_manager_instance(
                settings.GEMINI_API_KEYS, settings.VERTEX_API_KEYS
            )
        return self.key_manager

    async def initialize_upload(
        self,
        headers: Dict[str, str],
        body: Optional[bytes],
        user_token: str,
        request_host: Optional[str] = None,  # Add request host parameter
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Initialize file upload

        Args:
            headers: Request headers
            body: Request body
            user_token: User token

        Returns:
            Tuple[Dict[str, Any], Dict[str, str]]: (Response body, Response headers)
        """
        try:
            # Get an available API key
            key_manager = await self._get_key_manager()
            api_key = await key_manager.get_next_key()

            if not api_key:
                raise HTTPException(status_code=503, detail="No available API keys")

            # Forward the request to the real Gemini API
            async with AsyncClient() as client:
                # Prepare request headers
                forward_headers = {
                    "X-Goog-Upload-Protocol": headers.get(
                        "x-goog-upload-protocol", "resumable"
                    ),
                    "X-Goog-Upload-Command": headers.get(
                        "x-goog-upload-command", "start"
                    ),
                    "Content-Type": headers.get("content-type", "application/json"),
                }

                # Add other necessary headers
                if "x-goog-upload-header-content-length" in headers:
                    forward_headers["X-Goog-Upload-Header-Content-Length"] = headers[
                        "x-goog-upload-header-content-length"
                    ]
                if "x-goog-upload-header-content-type" in headers:
                    forward_headers["X-Goog-Upload-Header-Content-Type"] = headers[
                        "x-goog-upload-header-content-type"
                    ]

                # Send the request
                response = await client.post(
                    "https://generativelanguage.googleapis.com/upload/v1beta/files",
                    headers=forward_headers,
                    content=body,
                    params={"key": api_key},
                )

                if response.status_code != 200:
                    logger.error(
                        f"Upload initialization failed: {response.status_code} - {response.text}"
                    )
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Upload initialization failed",
                    )

                # Get the upload URL
                upload_url = response.headers.get("x-goog-upload-url")
                if not upload_url:
                    raise HTTPException(
                        status_code=500, detail="No upload URL in response"
                    )

                logger.info(f"Original upload URL from Google: {upload_url}")

                # Store upload information in headers for subsequent use
                # Do not create a database record here, wait until the upload is complete
                logger.info(
                    f"Upload initialized with API key: {redact_key_for_logging(api_key)}"
                )

                # Parse the response - the initialization response may be empty
                response_data = {}

                # Parse file information from the request body (if any)
                display_name = ""
                if body:
                    try:
                        request_data = json.loads(body)
                        display_name = request_data.get("displayName", "")
                    except Exception:
                        pass
                # Extract upload_id from the upload URL
                import urllib.parse

                parsed_url = urllib.parse.urlparse(upload_url)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                upload_id = query_params.get("upload_id", [None])[0]

                if upload_id:
                    # Store upload session information, using upload_id as the key
                    async with _upload_sessions_lock:
                        _upload_sessions[upload_id] = {
                            "api_key": api_key,
                            "user_token": user_token,
                            "display_name": display_name,
                            "mime_type": headers.get(
                                "x-goog-upload-header-content-type",
                                "application/octet-stream",
                            ),
                            "size_bytes": int(
                                headers.get("x-goog-upload-header-content-length", "0")
                            ),
                            "created_at": datetime.now(timezone.utc),
                            "upload_url": upload_url,
                        }
                        logger.info(
                            f"Stored upload session for upload_id={upload_id}: api_key={redact_key_for_logging(api_key)}"
                        )
                        logger.debug(f"Total active sessions: {len(_upload_sessions)}")
                else:
                    logger.warning(f"No upload_id found in upload URL: {upload_url}")

                # Periodically clean up expired sessions (older than 1 hour)
                asyncio.create_task(self._cleanup_expired_sessions())

                # Replace Google's URL with our proxy URL
                proxy_upload_url = upload_url
                if request_host:
                    # Original: https://generativelanguage.googleapis.com/upload/v1beta/files?key=AIzaSyDc...&upload_id=xxx&upload_protocol=resumable
                    # Replace with: http://request-host/upload/v1beta/files?key=sk-123456&upload_id=xxx&upload_protocol=resumable

                    # First, replace the domain
                    proxy_upload_url = upload_url.replace(
                        "https://generativelanguage.googleapis.com",
                        request_host.rstrip("/"),
                    )

                    # Then, replace the key parameter
                    import re

                    # Match the key=xxx parameter
                    key_pattern = r"(\?|&)key=([^&]+)"
                    match = re.search(key_pattern, proxy_upload_url)
                    if match:
                        # Replace with our token
                        proxy_upload_url = proxy_upload_url.replace(
                            f"{match.group(1)}key={match.group(2)}",
                            f"{match.group(1)}key={user_token}",
                        )

                    logger.info(
                        f"Replaced upload URL: {upload_url} -> {proxy_upload_url}"
                    )

                return response_data, {
                    "X-Goog-Upload-URL": proxy_upload_url,
                    "X-Goog-Upload-Status": "active",
                }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize upload: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    async def _cleanup_expired_sessions(self):
        """Clean up expired upload sessions"""
        try:
            async with _upload_sessions_lock:
                now = datetime.now(timezone.utc)
                expired_keys = []
                for key, session in _upload_sessions.items():
                    if now - session["created_at"] > timedelta(hours=1):
                        expired_keys.append(key)

                for key in expired_keys:
                    del _upload_sessions[key]

                if expired_keys:
                    logger.info(
                        f"Cleaned up {len(expired_keys)} expired upload sessions"
                    )
        except Exception as e:
            logger.error(f"Error cleaning up upload sessions: {str(e)}")

    async def get_upload_session(self, key: str) -> Optional[Dict[str, Any]]:
        """Get upload session information (supports upload_id or full URL)"""
        async with _upload_sessions_lock:
            # First, try to find directly
            session = _upload_sessions.get(key)
            if session:
                logger.debug(
                    f"Found session by direct key {redact_key_for_logging(key)}"
                )
                return session

            # If it's a URL, try to extract upload_id
            if key.startswith("http"):
                import urllib.parse

                parsed_url = urllib.parse.urlparse(key)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                upload_id = query_params.get("upload_id", [None])[0]
                if upload_id:
                    session = _upload_sessions.get(upload_id)
                    if session:
                        logger.debug(f"Found session by upload_id {upload_id} from URL")
                        return session

            logger.debug(f"No session found for key: {redact_key_for_logging(key)}")
            return None

    async def get_file(self, file_name: str, user_token: str) -> FileMetadata:
        """
        Get file information

        Args:
            file_name: File name (format: files/{file_id})
            user_token: User token

        Returns:
            FileMetadata: File metadata
        """
        try:
            # Query the file record
            file_record = await db_services.get_file_record_by_name(file_name)

            if not file_record:
                raise HTTPException(status_code=404, detail="File not found")

            # Check if it's expired
            expiration_time = datetime.fromisoformat(
                str(file_record["expiration_time"])
            )
            # If it's a naive datetime, assume UTC
            if expiration_time.tzinfo is None:
                expiration_time = expiration_time.replace(tzinfo=timezone.utc)
            if expiration_time <= datetime.now(timezone.utc):
                raise HTTPException(status_code=404, detail="File has expired")

            # Use the original API key to get file information
            api_key = file_record["api_key"]

            async with AsyncClient() as client:
                response = await client.get(
                    f"{settings.BASE_URL}/{file_name}", params={"key": api_key}
                )

                if response.status_code != 200:
                    logger.error(
                        f"Failed to get file: {response.status_code} - {response.text}"
                    )
                    raise HTTPException(
                        status_code=response.status_code, detail="Failed to get file"
                    )

                file_data = response.json()

                # Check and update the file state
                google_state = file_data.get("state", "PROCESSING")
                if (
                    google_state != file_record.get("state", "").value
                    if file_record.get("state")
                    else None
                ):
                    logger.info(
                        f"File state changed from {file_record.get('state')} to {google_state}"
                    )
                    # Update the state in the database
                    if google_state == "ACTIVE":
                        await db_services.update_file_record_state(
                            file_name=file_name,
                            state=FileState.ACTIVE,
                            update_time=datetime.now(timezone.utc),
                        )
                    elif google_state == "FAILED":
                        await db_services.update_file_record_state(
                            file_name=file_name,
                            state=FileState.FAILED,
                            update_time=datetime.now(timezone.utc),
                        )

                # Build the response
                return FileMetadata(
                    name=file_data["name"],
                    displayName=file_data.get("displayName"),
                    mimeType=file_data["mimeType"],
                    sizeBytes=str(file_data["sizeBytes"]),
                    createTime=file_data["createTime"],
                    updateTime=file_data["updateTime"],
                    expirationTime=file_data["expirationTime"],
                    sha256Hash=file_data.get("sha256Hash"),
                    uri=file_data["uri"],
                    state=google_state,
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get file {file_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    async def list_files(
        self,
        page_size: int = 10,
        page_token: Optional[str] = None,
        user_token: Optional[str] = None,
    ) -> ListFilesResponse:
        """
        List files

        Args:
            page_size: Page size
            page_token: Page token
            user_token: User token (optional, if provided, only returns files for that user)

        Returns:
            ListFilesResponse: File list response
        """
        try:
            logger.debug(
                f"list_files called with page_size={page_size}, page_token={page_token}"
            )

            # Get the file list from the database
            files, next_page_token = await db_services.list_file_records(
                user_token=user_token, page_size=page_size, page_token=page_token
            )

            logger.debug(
                f"Database returned {len(files)} files, next_page_token={next_page_token}"
            )

            # Convert to the response format
            file_list = []
            for file_record in files:
                file_list.append(
                    FileMetadata(
                        name=file_record["name"],
                        displayName=file_record.get("display_name"),
                        mimeType=file_record["mime_type"],
                        sizeBytes=str(file_record["size_bytes"]),
                        createTime=file_record["create_time"].isoformat() + "Z",
                        updateTime=file_record["update_time"].isoformat() + "Z",
                        expirationTime=file_record["expiration_time"].isoformat() + "Z",
                        sha256Hash=file_record.get("sha256_hash"),
                        uri=file_record["uri"],
                        state=file_record["state"].value
                        if file_record.get("state")
                        else "ACTIVE",
                    )
                )

            response = ListFilesResponse(files=file_list, nextPageToken=next_page_token)

            logger.debug(
                f"Returning response with {len(response.files)} files, nextPageToken={response.nextPageToken}"
            )

            return response

        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    async def delete_file(self, file_name: str, user_token: str) -> bool:
        """
        Delete a file

        Args:
            file_name: File name
            user_token: User token

        Returns:
            bool: Whether the deletion was successful
        """
        try:
            # Query the file record
            file_record = await db_services.get_file_record_by_name(file_name)

            if not file_record:
                raise HTTPException(status_code=404, detail="File not found")

            # Use the original API key to delete the file
            api_key = file_record["api_key"]

            async with AsyncClient() as client:
                response = await client.delete(
                    f"{settings.BASE_URL}/{file_name}", params={"key": api_key}
                )

                if response.status_code not in [200, 204]:
                    logger.error(
                        f"Failed to delete file: {response.status_code} - {response.text}"
                    )
                    # If the API deletion fails, but the file has expired, still delete the database record
                    expiration_time = datetime.fromisoformat(
                        str(file_record["expiration_time"])
                    )
                    if expiration_time.tzinfo is None:
                        expiration_time = expiration_time.replace(tzinfo=timezone.utc)
                    if expiration_time <= datetime.now(timezone.utc):
                        await db_services.delete_file_record(file_name)
                        return True
                    raise HTTPException(
                        status_code=response.status_code, detail="Failed to delete file"
                    )

            # Delete the database record
            await db_services.delete_file_record(file_name)
            return True

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete file {file_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    async def check_file_state(self, file_name: str, api_key: str) -> str:
        """
        Check and update the file state

        Args:
            file_name: File name
            api_key: API key

        Returns:
            str: Current state
        """
        try:
            async with AsyncClient() as client:
                response = await client.get(
                    f"{settings.BASE_URL}/{file_name}", params={"key": api_key}
                )

                if response.status_code != 200:
                    logger.error(f"Failed to check file state: {response.status_code}")
                    return "UNKNOWN"

                file_data = response.json()
                google_state = file_data.get("state", "PROCESSING")

                # Update the database state
                if google_state == "ACTIVE":
                    await db_services.update_file_record_state(
                        file_name=file_name,
                        state=FileState.ACTIVE,
                        update_time=datetime.now(timezone.utc),
                    )
                elif google_state == "FAILED":
                    await db_services.update_file_record_state(
                        file_name=file_name,
                        state=FileState.FAILED,
                        update_time=datetime.now(timezone.utc),
                    )

                return google_state

        except Exception as e:
            logger.error(f"Failed to check file state: {str(e)}")
            return "UNKNOWN"

    async def cleanup_expired_files(self) -> int:
        """
        Clean up expired files

        Returns:
            int: Number of cleaned up files
        """
        try:
            # Get expired files
            expired_files = await db_services.delete_expired_file_records()

            if not expired_files:
                return 0

            # Try to delete files from the Gemini API
            for file_record in expired_files:
                try:
                    api_key = file_record["api_key"]
                    file_name = file_record["name"]

                    async with AsyncClient() as client:
                        await client.delete(
                            f"{settings.BASE_URL}/{file_name}", params={"key": api_key}
                        )
                except Exception as e:
                    # Log the error but continue processing other files
                    logger.error(
                        f"Failed to delete file {file_record['name']} from API: {str(e)}"
                    )

            return len(expired_files)

        except Exception as e:
            logger.error(f"Failed to cleanup expired files: {str(e)}")
            return 0


# Singleton instance
_files_service_instance: Optional[FilesService] = None


async def get_files_service() -> FilesService:
    """Get the file service singleton instance"""
    global _files_service_instance
    if _files_service_instance is None:
        _files_service_instance = FilesService()
    return _files_service_instance
