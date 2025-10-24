"""
Domain models related to the Files API.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class FileUploadConfig(BaseModel):
    """File upload configuration."""

    mime_type: Optional[str] = Field(None, description="MIME type")
    display_name: Optional[str] = Field(
        None, description="Display name, up to 40 characters"
    )


class CreateFileRequest(BaseModel):
    """Create file request (for initializing an upload)."""

    file: Optional[Dict[str, Any]] = Field(None, description="File metadata")


class FileMetadata(BaseModel):
    """File metadata response."""

    name: str = Field(..., description="File name, format: files/{file_id}")
    displayName: Optional[str] = Field(None, description="Display name")
    mimeType: str = Field(..., description="MIME type")
    sizeBytes: str = Field(..., description="File size (bytes)")
    createTime: str = Field(..., description="Creation time (RFC3339)")
    updateTime: str = Field(..., description="Update time (RFC3339)")
    expirationTime: str = Field(..., description="Expiration time (RFC3339)")
    sha256Hash: Optional[str] = Field(None, description="SHA256 hash value")
    uri: str = Field(..., description="File access URI")
    state: str = Field(..., description="File status")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() + "Z"}


class ListFilesRequest(BaseModel):
    """List files request parameters."""

    pageSize: Optional[int] = Field(10, ge=1, le=100, description="Page size")
    pageToken: Optional[str] = Field(None, description="Page token")


class ListFilesResponse(BaseModel):
    """List files response."""

    files: List[FileMetadata] = Field(default_factory=list, description="List of files")
    nextPageToken: Optional[str] = Field(None, description="Next page token")


class UploadInitResponse(BaseModel):
    """Upload initialization response (internal use)."""

    file_metadata: FileMetadata
    upload_url: str


class FileKeyMapping(BaseModel):
    """Mapping between file and API Key (internal use)."""

    file_name: str
    api_key: str
    user_token: str
    created_at: datetime
    expires_at: datetime


class DeleteFileResponse(BaseModel):
    """Delete file response."""

    success: bool = Field(..., description="Whether the deletion was successful")
    message: Optional[str] = Field(None, description="Message")
