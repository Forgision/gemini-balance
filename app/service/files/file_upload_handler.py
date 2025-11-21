"""
File Upload Handler
Handles Google's resumable upload protocol
"""

from typing import Optional
from datetime import datetime, timezone, timedelta

from httpx import AsyncClient
from fastapi import Request, Response, HTTPException

from app.config.config import settings
from app.database import services as db_services
from app.database.models import FileState
from app.log.logger import get_files_logger
from app.utils.helpers import redact_key_for_logging

logger = get_files_logger()


class FileUploadHandler:
    """Handles file chunk uploading"""

    def __init__(self):
        self.chunk_size = 8 * 1024 * 1024  # 8MB

    async def handle_upload_chunk(
        self,
        upload_url: str,
        request: Request,
        files_service=None,  # Add files_service parameter
    ) -> Response:
        """
        Handle upload chunk

        Args:
            upload_url: Upload URL
            request: FastAPI request object
            files_service: File service instance

        Returns:
            Response: Response object
        """
        try:
            # Get request headers
            headers = {}

            # Copy necessary upload headers
            upload_headers = [
                "x-goog-upload-command",
                "x-goog-upload-offset",
                "content-type",
                "content-length",
            ]

            for header in upload_headers:
                if header in request.headers:
                    # Convert to the correct format
                    key = "-".join(word.capitalize() for word in header.split("-"))
                    headers[key] = request.headers[header]

            # Read the request body
            body = await request.body()

            # Check if it's the last chunk
            is_final = "finalize" in headers.get("X-Goog-Upload-Command", "")
            logger.debug(
                f"Upload command: {headers.get('X-Goog-Upload-Command', '')}, is_final: {is_final}"
            )

            # Forward to the real upload URL
            async with AsyncClient() as client:
                response = await client.post(
                    upload_url,
                    headers=headers,
                    content=body,
                    timeout=300.0,  # 5 minutes timeout
                )

                if response.status_code not in [200, 201, 308]:
                    logger.error(
                        f"Upload chunk failed: {response.status_code} - {response.text}"
                    )
                    raise HTTPException(
                        status_code=response.status_code, detail="Upload failed"
                    )

                # If it's the last chunk, update the file state
                if is_final and response.status_code in [200, 201]:
                    logger.debug(f"Upload finalized with status {response.status_code}")
                    try:
                        # Parse the response to get file information
                        response_data = response.json()
                        logger.debug(f"Upload complete response data: {response_data}")
                        file_data = response_data.get("file", {})

                        # Get the real file name
                        real_file_name = file_data.get("name")
                        logger.debug(f"Upload response: {response_data}")
                        if real_file_name and files_service:
                            logger.info(
                                f"Upload completed, file name: {real_file_name}"
                            )

                            # Get information from the session
                            session_info = await files_service.get_upload_session(
                                upload_url
                            )
                            logger.debug(
                                f"Retrieved session info for {upload_url}: {session_info}"
                            )
                            if session_info:
                                # Create a file record
                                now = datetime.now(timezone.utc)
                                expiration_time = now + timedelta(hours=48)

                                # Handle the expiration time format (Google may return nanosecond precision)
                                expiration_time_str = file_data.get(
                                    "expirationTime", expiration_time.isoformat() + "Z"
                                )
                                # Handle nanosecond format: 2025-07-11T02:02:52.531916141Z -> 2025-07-11T02:02:52.531916Z
                                if expiration_time_str.endswith("Z"):
                                    # Remove Z
                                    expiration_time_str = expiration_time_str[:-1]
                                    # If there are nanoseconds (more than 6 decimal places), truncate to microseconds
                                    if "." in expiration_time_str:
                                        date_part, frac_part = (
                                            expiration_time_str.rsplit(".", 1)
                                        )
                                        if len(frac_part) > 6:
                                            frac_part = frac_part[:6]
                                        expiration_time_str = f"{date_part}.{frac_part}"
                                    # Add timezone
                                    expiration_time_str += "+00:00"

                                # Get the file state (Google may return PROCESSING)
                                file_state = file_data.get("state", "PROCESSING")
                                logger.debug(f"File state from Google: {file_state}")

                                # Convert the string state to an enum
                                if file_state == "ACTIVE":
                                    state_enum = FileState.ACTIVE
                                elif file_state == "PROCESSING":
                                    state_enum = FileState.PROCESSING
                                elif file_state == "FAILED":
                                    state_enum = FileState.FAILED
                                else:
                                    logger.warning(
                                        f"Unknown file state: {file_state}, defaulting to PROCESSING"
                                    )
                                    state_enum = FileState.PROCESSING

                                from app.database.connection import AsyncSessionLocal
                                async with AsyncSessionLocal() as session:
                                    await db_services.create_file_record(
                                        session,
                                        name=real_file_name,
                                        mime_type=file_data.get(
                                            "mimeType", session_info["mime_type"]
                                        ),
                                        size_bytes=int(
                                            file_data.get(
                                                "sizeBytes", session_info["size_bytes"]
                                            )
                                        ),
                                        api_key=session_info["api_key"],
                                        uri=file_data.get(
                                            "uri", f"{settings.BASE_URL}/{real_file_name}"
                                        ),
                                        create_time=now,
                                        update_time=now,
                                        expiration_time=datetime.fromisoformat(
                                            expiration_time_str
                                        ),
                                        state=state_enum,
                                        display_name=file_data.get(
                                            "displayName",
                                            session_info.get("display_name", ""),
                                        ),
                                        sha256_hash=file_data.get("sha256Hash"),
                                        user_token=session_info["user_token"],
                                    )
                                logger.info(
                                    f"Created file record: name={real_file_name}, api_key={redact_key_for_logging(session_info['api_key'])}"
                                )
                            else:
                                logger.warning(
                                    f"No upload session found for URL: {upload_url}"
                                )
                        else:
                            logger.warning(
                                f"Missing real_file_name or files_service: real_file_name={real_file_name}, files_service={files_service}"
                            )

                        # Return the complete file information
                        return Response(
                            content=response.content,
                            status_code=response.status_code,
                            headers=dict(response.headers),
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to create file record: {str(e)}", exc_info=True
                        )
                else:
                    logger.debug(
                        f"Upload chunk processed: is_final={is_final}, status={response.status_code}"
                    )

                # Return the response
                response_headers = dict(response.headers)

                # Ensure necessary headers are included
                if response.status_code == 308:  # Resume Incomplete
                    if "x-goog-upload-status" not in response_headers:
                        response_headers["x-goog-upload-status"] = "active"

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to handle upload chunk: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    async def proxy_upload_request(
        self, request: Request, upload_url: str, files_service=None
    ) -> Response:
        """
        Proxy the upload request

        Args:
            request: FastAPI request object
            upload_url: Target upload URL
            files_service: File service instance

        Returns:
            Response: Proxied response
        """
        logger.debug(f"Proxy upload request: {request.method}, {upload_url}")
        try:
            # If it's a GET request, return the upload status
            if request.method == "GET":
                return await self._get_upload_status(upload_url)

            # Handle POST/PUT requests
            return await self.handle_upload_chunk(upload_url, request, files_service)

        except Exception as e:
            logger.error(f"Failed to proxy upload request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    async def _get_upload_status(self, upload_url: str) -> Response:
        """
        Get upload status

        Args:
            upload_url: Upload URL

        Returns:
            Response: Status response
        """
        try:
            async with AsyncClient() as client:
                response = await client.get(upload_url)

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
        except Exception as e:
            logger.error(f"Failed to get upload status: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# Singleton instance
_upload_handler_instance: Optional[FileUploadHandler] = None


def get_upload_handler() -> FileUploadHandler:
    """Get the upload handler singleton instance"""
    global _upload_handler_instance
    if _upload_handler_instance is None:
        _upload_handler_instance = FileUploadHandler()
    return _upload_handler_instance
