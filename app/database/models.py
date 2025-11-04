"""
Database Models Module
"""

import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    JSON,
    Boolean,
    BigInteger,
    Enum,
)
import enum

from app.database.connection import Base


class Settings(Base):
    """
    Settings table, corresponding to the configuration items in .env
    """

    __tablename__ = "t_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(
        String(100), nullable=False, unique=True, comment="Configuration item key name"
    )
    value = Column(Text, nullable=True, comment="Configuration item value")
    description = Column(
        String(255), nullable=True, comment="Configuration item description"
    )
    created_at = Column(
        DateTime, default=datetime.datetime.now, comment="Creation time"
    )
    updated_at = Column(
        DateTime,
        default=datetime.datetime.now,
        onupdate=datetime.datetime.now,
        comment="Update time",
    )

    def __repr__(self):
        return f"<Settings(key='{self.key}', value='{self.value}')>"


class ErrorLog(Base):
    """
    Error log table
    """

    __tablename__ = "t_error_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    gemini_key = Column(String(100), nullable=True, comment="Gemini API key")
    model_name = Column(String(100), nullable=True, comment="Model name")
    error_type = Column(String(50), nullable=True, comment="Error type")
    error_log = Column(Text, nullable=True, comment="Error log")
    error_code = Column(Integer, nullable=True, comment="Error code")
    request_msg = Column(JSON, nullable=True, comment="Request message")
    request_time = Column(
        DateTime, default=datetime.datetime.now, comment="Request time"
    )

    def __repr__(self):
        return f"<ErrorLog(id='{self.id}', gemini_key='{self.gemini_key}')>"


class RequestLog(Base):
    """
    API request log table
    """

    __tablename__ = "t_request_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_time = Column(
        DateTime, default=datetime.datetime.now, comment="Request time"
    )
    model_name = Column(String(100), nullable=True, comment="Model name")
    api_key = Column(String(100), nullable=True, comment="API key used")
    is_success = Column(
        Boolean, nullable=False, comment="Whether the request was successful"
    )
    status_code = Column(Integer, nullable=True, comment="API response status code")
    latency_ms = Column(
        Integer, nullable=True, comment="Request latency (milliseconds)"
    )
    token_count = Column(Integer, nullable=True, comment="Token count for the request")

    def __repr__(self):
        return f"<RequestLog(id='{self.id}', key='{self.api_key[:4]}...', success='{self.is_success}')>"


class FileState(enum.Enum):
    """File status enumeration"""

    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class FileRecord(Base):
    """
    File record table for storing information about files uploaded to Gemini
    """

    __tablename__ = "t_file_records"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # File basic information
    name = Column(
        String(255),
        unique=True,
        nullable=False,
        comment="File name, format: files/{file_id}",
    )
    display_name = Column(
        String(255),
        nullable=True,
        comment="Original file name when uploaded by the user",
    )
    mime_type = Column(String(100), nullable=False, comment="MIME type")
    size_bytes = Column(BigInteger, nullable=False, comment="File size (bytes)")
    sha256_hash = Column(String(255), nullable=True, comment="SHA256 hash of the file")

    # Status information
    state = Column(
        Enum(FileState),
        nullable=False,
        default=FileState.PROCESSING,
        comment="File status",
    )

    # Timestamps
    create_time = Column(DateTime, nullable=False, comment="Creation time")
    update_time = Column(DateTime, nullable=False, comment="Update time")
    expiration_time = Column(DateTime, nullable=False, comment="Expiration time")

    # API related
    uri = Column(String(500), nullable=False, comment="File access URI")
    api_key = Column(String(100), nullable=False, comment="API Key used for upload")
    upload_url = Column(
        Text, nullable=True, comment="Temporary upload URL (for chunked uploads)"
    )

    # Additional information
    user_token = Column(
        String(100), nullable=True, comment="Token of the uploading user"
    )
    upload_completed = Column(DateTime, nullable=True, comment="Upload completion time")

    def __repr__(self):
        return f"<FileRecord(name='{self.name}', state='{self.state.value if self.state is not None else 'None'}', api_key='{self.api_key[:8]}...')>"

    def to_dict(self):
        """Convert to dictionary format for API response"""
        return {
            "name": self.name,
            "displayName": self.display_name,
            "mimeType": self.mime_type,
            "sizeBytes": str(self.size_bytes),
            "createTime": self.create_time.isoformat() + "Z",
            "updateTime": self.update_time.isoformat() + "Z",
            "expirationTime": self.expiration_time.isoformat() + "Z",
            "sha256Hash": self.sha256_hash,
            "uri": self.uri,
            "state": self.state.value if self.state is not None else "PROCESSING",
        }

    def is_expired(self):
        """Check if the file has expired"""
        # Ensure comparison is timezone-aware
        expiration_time = self.expiration_time
        if expiration_time.tzinfo is None:
            expiration_time = expiration_time.replace(tzinfo=datetime.timezone.utc)
        return datetime.datetime.now(datetime.timezone.utc) > expiration_time


class UsageStats(Base):
    """
    Usage statistics table
    """

    __tablename__ = "t_usage_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    api_key = Column(String(100), nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    token_count = Column(Integer, nullable=False, default=0)
    rpm = Column(Integer, nullable=False, default=0)
    rpd = Column(Integer, nullable=False, default=0)
    tpm = Column(Integer, nullable=False, default=0)
    exhausted = Column(Boolean, nullable=False, default=False)
    rpm_timestamp = Column(DateTime, nullable=True)
    tpm_timestamp = Column(DateTime, nullable=True)
    rpd_timestamp = Column(DateTime, nullable=True)
    timestamp = Column(
        DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now
    )

    def __repr__(self):
        return (
            f"<UsageStats(api_key='{self.api_key}', "
            f"model_name='{self.model_name}', "
            f"token_count='{self.token_count}'"
            f"rpm='{self.rpm}', "
            f"rpm_timestamp='{self.rpm_timestamp}', "
            f"rpd='{self.rpd}', "
            f"rpd_timestamp='{self.rpd_timestamp}', "
            f"tpm='{self.tpm}', "
            f"tpm_timestamp='{self.tpm_timestamp}', "
            f"exhausted='{self.exhausted}')>"
        )
