"""
Files API routes
"""

from typing import Optional, Union
from fastapi import APIRouter, Request, Query, Depends, Header, HTTPException
from fastapi.responses import JSONResponse

from app.config.config import settings
from app.dependencies import get_files_service
from app.domain.file_models import FileMetadata, ListFilesResponse, DeleteFileResponse
from app.log.logger import get_files_logger
from app.core.security import SecurityService
from app.service.files.file_upload_handler import get_upload_handler
from app.utils.helpers import redact_key_for_logging

logger = get_files_logger()

router = APIRouter()
security_service = SecurityService()


@router.post("/upload/v1beta/files")
async def upload_file_init(
    request: Request,
    auth_token: str = Depends(security_service.verify_key_or_goog_api_key),
    x_goog_upload_protocol: Optional[str] = Header(None),
    x_goog_upload_command: Optional[str] = Header(None),
    x_goog_upload_header_content_length: Optional[str] = Header(None),
    x_goog_upload_header_content_type: Optional[str] = Header(None),
    files_service=Depends(get_files_service),
):
    """Initialize file upload"""
    logger.debug(
        f"Upload file request: {request.method=}, {request.url=}, {auth_token=}, {x_goog_upload_protocol=}, {x_goog_upload_command=}, {x_goog_upload_header_content_length=}, {x_goog_upload_header_content_type=}"
    )

    # Check if it is an actual upload request (with upload_id)
    if request.query_params.get("upload_id") and x_goog_upload_command in [
        "upload",
        "upload, finalize",
    ]:
        logger.debug(
            "This is an upload request, not initialization. Redirecting to handle_upload."
        )
        return await handle_upload(
            upload_path="v1beta/files",
            request=request,
            key=request.query_params.get("key"),
            auth_token=auth_token,
            files_service=files_service,
        )

    try:
        # Use the authentication token as user_token
        user_token = auth_token
        # Get the request body
        body = await request.body()

        # Build the request host URL
        request_host = f"{request.url.scheme}://{request.url.netloc}"
        logger.info(f"Request host: {request_host}")

        # Prepare request headers
        headers = {
            "x-goog-upload-protocol": x_goog_upload_protocol or "resumable",
            "x-goog-upload-command": x_goog_upload_command or "start",
        }

        if x_goog_upload_header_content_length:
            headers["x-goog-upload-header-content-length"] = (
                x_goog_upload_header_content_length
            )
        if x_goog_upload_header_content_type:
            headers["x-goog-upload-header-content-type"] = (
                x_goog_upload_header_content_type
            )

        # Call the service
        response_data, response_headers = await files_service.initialize_upload(
            headers=headers,
            body=body,
            user_token=user_token,
            request_host=request_host,  # Pass the request host
            request=request,  # Pass request for app.state access
        )

        logger.info(f"Upload initialization response: {response_data}")
        logger.info(f"Upload initialization response headers: {response_headers}")

        logger.info(f"Upload initialization response headers: {response_data}")
        # Return the response
        return JSONResponse(content=response_data, headers=response_headers)

    except HTTPException as e:
        logger.error(f"Upload initialization failed: {e.detail}")
        return JSONResponse(
            content={"error": {"message": e.detail}}, status_code=e.status_code
        )
    except Exception as e:
        logger.error(f"Unexpected error in upload initialization: {str(e)}")
        return JSONResponse(
            content={"error": {"message": "Internal server error"}}, status_code=500
        )


@router.get("/v1beta/files", response_model=ListFilesResponse)
async def list_files(
    page_size: int = Query(10, ge=1, le=100, description="Page size", alias="pageSize"),
    page_token: Optional[str] = Query(
        None, description="Page token", alias="pageToken"
    ),
    auth_token: str = Depends(security_service.verify_key_or_goog_api_key),
    files_service=Depends(get_files_service),
) -> Union[ListFilesResponse, JSONResponse]:
    """List files"""
    logger.debug(f"List files: {page_size=}, {page_token=}, {auth_token=}")
    try:
        # Use the authentication token as user_token (if user isolation is enabled)
        user_token = auth_token if settings.FILES_USER_ISOLATION_ENABLED else None
        # Call the service
        return await files_service.list_files(
            page_size=page_size, page_token=page_token, user_token=user_token
        )

    except HTTPException as e:
        logger.error(f"List files failed: {e.detail}")
        return JSONResponse(
            content={"error": {"message": e.detail}}, status_code=e.status_code
        )
    except Exception as e:
        logger.error(f"Unexpected error in list files: {str(e)}")
        return JSONResponse(
            content={"error": {"message": "Internal server error"}}, status_code=500
        )


@router.get("/v1beta/files/{file_id:path}", response_model=FileMetadata)
async def get_file(
    file_id: str,
    auth_token: str = Depends(security_service.verify_key_or_goog_api_key),
    files_service=Depends(get_files_service),
) -> Union[FileMetadata, JSONResponse]:
    """Get file information"""
    logger.debug(f"Get file request: {file_id=}, {auth_token=}")
    try:
        # Use the authentication token as user_token
        user_token = auth_token
        # Call the service
        return await files_service.get_file(f"files/{file_id}", user_token)

    except HTTPException as e:
        logger.error(f"Get file failed: {e.detail}")
        return JSONResponse(
            content={"error": {"message": e.detail}}, status_code=e.status_code
        )
    except Exception as e:
        logger.error(f"Unexpected error in get file: {str(e)}")
        return JSONResponse(
            content={"error": {"message": "Internal server error"}}, status_code=500
        )


@router.delete("/v1beta/files/{file_id:path}", response_model=DeleteFileResponse)
async def delete_file(
    file_id: str,
    auth_token: str = Depends(security_service.verify_key_or_goog_api_key),
    files_service=Depends(get_files_service),
) -> Union[DeleteFileResponse, JSONResponse]:
    """Delete file"""
    logger.info(f"Delete file: {file_id=}, {auth_token=}")
    try:
        # Use the authentication token as user_token
        user_token = auth_token
        # Call the service
        success = await files_service.delete_file(f"files/{file_id}", user_token)

        return DeleteFileResponse(
            success=success,
            message="File deleted successfully" if success else "Failed to delete file",
        )

    except HTTPException as e:
        logger.error(f"Delete file failed: {e.detail}")
        return JSONResponse(
            content={"error": {"message": e.detail}}, status_code=e.status_code
        )
    except Exception as e:
        logger.error(f"Unexpected error in delete file: {str(e)}")
        return JSONResponse(
            content={"error": {"message": "Internal server error"}}, status_code=500
        )


# Wildcard route for handling upload requests
@router.api_route("/upload/{upload_path:path}", methods=["GET", "POST", "PUT"])
async def handle_upload(
    upload_path: str,
    request: Request,
    key: Optional[str] = Query(None),  # Get key from query parameters
    auth_token: str = Depends(security_service.verify_key_or_goog_api_key),
    files_service=Depends(get_files_service),
):
    """Handle file upload requests"""
    try:
        logger.info(
            f"Handling upload request: {request.method} {upload_path}, key={redact_key_for_logging(key if key is not None else 'N/A')}"
        )

        # Get upload_id from query parameters
        upload_id = request.query_params.get("upload_id")
        if not upload_id:
            raise HTTPException(status_code=400, detail="Missing upload_id")

        # Get the real API key from the session
        session_info = await files_service.get_upload_session(upload_id)
        if not session_info:
            logger.error(f"No session found for upload_id: {upload_id}")
            raise HTTPException(status_code=404, detail="Upload session not found")

        real_api_key = session_info["api_key"]
        original_upload_url = session_info["upload_url"]

        # Use the real API key to build the complete Google upload URL
        # Keep all parameters of the original URL, but use the real API key
        upload_url = original_upload_url
        logger.info(
            f"Using real API key for upload: {redact_key_for_logging(real_api_key)}"
        )

        # Proxy the upload request
        upload_handler = get_upload_handler()
        return await upload_handler.proxy_upload_request(
            request=request, upload_url=upload_url, files_service=files_service
        )

    except HTTPException as e:
        logger.error(f"Upload handling failed: {e.detail}")
        return JSONResponse(
            content={"error": {"message": e.detail}}, status_code=e.status_code
        )
    except Exception as e:
        logger.error(f"Unexpected error in upload handling: {str(e)}")
        return JSONResponse(
            content={"error": {"message": "Internal server error"}}, status_code=500
        )


# Add /gemini prefix route for compatibility
@router.post("/gemini/upload/v1beta/files")
async def gemini_upload_file_init(
    request: Request,
    auth_token: str = Depends(security_service.verify_key_or_goog_api_key),
    x_goog_upload_protocol: Optional[str] = Header(None),
    x_goog_upload_command: Optional[str] = Header(None),
    x_goog_upload_header_content_length: Optional[str] = Header(None),
    x_goog_upload_header_content_type: Optional[str] = Header(None),
    files_service=Depends(get_files_service),
):
    """Initialize file upload (Gemini prefix)"""
    return await upload_file_init(
        request,
        auth_token,
        x_goog_upload_protocol,
        x_goog_upload_command,
        x_goog_upload_header_content_length,
        x_goog_upload_header_content_type,
        files_service,
    )


@router.get("/gemini/v1beta/files", response_model=ListFilesResponse)
async def gemini_list_files(
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
    page_token: Optional[str] = Query(None, alias="pageToken"),
    auth_token: str = Depends(security_service.verify_key_or_goog_api_key),
    files_service=Depends(get_files_service),
) -> Union[ListFilesResponse, JSONResponse]:
    """List files (Gemini prefix)"""
    return await list_files(page_size, page_token, auth_token, files_service)


@router.get("/gemini/v1beta/files/{file_id:path}", response_model=FileMetadata)
async def gemini_get_file(
    file_id: str,
    auth_token: str = Depends(security_service.verify_key_or_goog_api_key),
    files_service=Depends(get_files_service),
) -> Union[FileMetadata, JSONResponse]:
    """Get file information (Gemini prefix)"""
    return await get_file(file_id, auth_token, files_service)


@router.delete("/gemini/v1beta/files/{file_id:path}", response_model=DeleteFileResponse)
async def gemini_delete_file(
    file_id: str,
    auth_token: str = Depends(security_service.verify_key_or_goog_api_key),
    files_service=Depends(get_files_service),
) -> Union[DeleteFileResponse, JSONResponse]:
    """Delete file (Gemini prefix)"""
    return await delete_file(file_id, auth_token, files_service)