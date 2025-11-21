"""
Database service module
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import asc, delete, desc, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ErrorLog,
    FileRecord,
    FileState,
    RequestLog,
    Settings,
    UsageStats,
)
from app.log.logger import get_database_logger
from app.utils.helpers import redact_key_for_logging

logger = get_database_logger()


async def get_all_settings(session: AsyncSession) -> List[Dict[str, Any]]:
    """
    Get all settings

    Args:
        session: Database session

    Returns:
        List[Dict[str, Any]]: List of settings
    """
    try:
        query = select(Settings)
        result = await session.execute(query)
        rows = result.scalars().all()
        return [dict(row.__dict__) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get all settings: {str(e)}", exc_info=True)
        raise


async def get_setting(session: AsyncSession, key: str) -> Optional[Dict[str, Any]]:
    """
    Get the setting for the specified key

    Args:
        session: Database session
        key: Setting key name

    Returns:
        Optional[Dict[str, Any]]: Setting information, or None if it does not exist
    """
    try:
        query = select(Settings).where(Settings.key == key)
        result = await session.execute(query)
        row = result.scalar_one_or_none()
        return dict(row.__dict__) if row else None
    except Exception as e:
        logger.error(f"Failed to get setting {key}: {str(e)}", exc_info=True)
        raise


async def update_setting(
    session: AsyncSession, key: str, value: str, description: Optional[str] = None
) -> bool:
    """
    Update setting

    Args:
        session: Database session
        key: Setting key name
        value: Setting value
        description: Setting description

    Returns:
        bool: Whether the update was successful
    """
    try:
        # Check if the setting exists
        setting = await get_setting(session, key)

        if setting:
            # Update setting
            await session.execute(
                update(Settings)
                .where(Settings.key == key)
                .values(
                    value=value,
                    description=description if description else setting["description"],
                    updated_at=datetime.now(),
                )
            )
            await session.commit()
            logger.info(f"Updated setting: {key}")
            return True
        else:
            # Insert setting
            await session.execute(
                insert(Settings).values(
                    key=key,
                    value=value,
                    description=description,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            )
            await session.commit()
            logger.info(f"Inserted setting: {key}")
            return True
    except Exception as e:
        logger.error(f"Failed to update setting {key}: {str(e)}", exc_info=True)
        return False


async def get_usage_stats_by_key_and_model(
    session: AsyncSession, api_key: str, model_name: str
) -> Optional[Dict[str, Any]]:
    """
    Get usage statistics for a given API key and model.

    Args:
        session: Database session
        api_key: The API key.
        model_name: The model name.

    Returns:
        Optional[Dict[str, Any]]: The usage statistics, or None if not found.
    """
    try:
        query = select(UsageStats).where(
            UsageStats.api_key == api_key,
            UsageStats.model_name == model_name,
        )
        result = await session.execute(query)
        row = result.scalar_one_or_none()
        if not row:
            return None

        record = dict(row.__dict__)
        now = datetime.now()

        # Check if RPM and TPM need to be reset
        if (
            record["rpm_timestamp"]
            and (now - record["rpm_timestamp"]).total_seconds() > 60
        ):
            record["rpm"] = 0
            record["tpm"] = 0

        # Check if RPD needs to be reset (Pacific Time)
        pacific_tz = timezone(timedelta(hours=-7))
        if (
            record["rpd_timestamp"]
            and record["rpd_timestamp"].astimezone(pacific_tz).date()
            < now.astimezone(pacific_tz).date()
        ):
            record["rpd"] = 0

        return record
    except Exception as e:
        logger.error(f"Failed to get usage stats: {str(e)}", exc_info=True)
        raise


async def get_all_usage_stats(session: AsyncSession) -> List[Dict[str, Any]]:
    """
    Get all usage statistics.

    Args:
        session: Database session

    Returns:
        List[Dict[str, Any]]: A list of usage statistics.
    """
    try:
        query = select(UsageStats)
        result = await session.execute(query)
        rows = result.scalars().all()
        return [dict(row.__dict__) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get all usage stats: {str(e)}", exc_info=True)
        raise


async def add_error_log(
    session: AsyncSession,
    gemini_key: Optional[str] = None,
    model_name: Optional[str] = None,
    error_type: Optional[str] = None,
    error_log: Optional[str] = None,
    error_code: Optional[int] = None,
    request_msg: Optional[Union[Dict[str, Any], str]] = None,
    request_datetime: Optional[datetime] = None,
) -> bool:
    """
    Add error log

    Args:
        gemini_key: Gemini API key
        error_log: Error log
        error_code: Error code (e.g., HTTP status code)
        request_msg: Request message

    Returns:
        bool: Whether the addition was successful
    """
    try:
        if request_msg is None:
            request_msg_json = None
        else:
            # If request_msg is a dictionary, convert it to a JSON string
            if isinstance(request_msg, dict):
                request_msg_json = request_msg
            elif isinstance(request_msg, str):
                try:
                    request_msg_json = json.loads(request_msg)
                except json.JSONDecodeError:
                    request_msg_json = {"message": request_msg}
            else:
                request_msg_json = None

        # Insert error log
        query = insert(ErrorLog).values(
            gemini_key=gemini_key,
            error_type=error_type,
            error_log=error_log,
            model_name=model_name,
            error_code=error_code,
            request_msg=request_msg_json,
            request_time=(request_datetime if request_datetime else datetime.now()),
        )
        await session.execute(query)
        await session.commit()
        logger.info(
            f"Added error log for key: {redact_key_for_logging(gemini_key if gemini_key is not None else 'N/A')}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to add error log: {str(e)}", exc_info=True)
        return False


async def get_error_logs(
    session: AsyncSession,
    limit: int = 20,
    offset: int = 0,
    key_search: Optional[str] = None,
    error_search: Optional[str] = None,
    error_code_search: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    sort_by: str = "id",
    sort_order: str = "desc",
) -> List[Dict[str, Any]]:
    """
    Get error logs, with support for searching, date filtering, and sorting

    Args:
        limit (int): Limit number
        offset (int): Offset
        key_search (Optional[str]): Gemini key search term (fuzzy match)
        error_search (Optional[str]): Error type or log content search term (fuzzy match)
        error_code_search (Optional[str]): Error code search term (exact match)
        start_date (Optional[datetime]): Start date and time
        end_date (Optional[datetime]): End date and time
        sort_by (str): Sort field (e.g., 'id', 'request_time')
        sort_order (str): Sort order ('asc' or 'desc')

    Returns:
        List[Dict[str, Any]]: List of error logs
    """
    try:
        query = select(
            ErrorLog.id,
            ErrorLog.gemini_key,
            ErrorLog.model_name,
            ErrorLog.error_type,
            ErrorLog.error_log,
            ErrorLog.error_code,
            ErrorLog.request_time,
        )

        if key_search:
            query = query.where(ErrorLog.gemini_key.ilike(f"%{key_search}%"))
        if error_search:
            query = query.where(
                (ErrorLog.error_type.ilike(f"%{error_search}%"))
                | (ErrorLog.error_log.ilike(f"%{error_search}%"))
            )
        if start_date:
            query = query.where(ErrorLog.request_time >= start_date)
        if end_date:
            query = query.where(ErrorLog.request_time < end_date)
        if error_code_search:
            try:
                error_code_int = int(error_code_search)
                query = query.where(ErrorLog.error_code == error_code_int)
            except ValueError:
                logger.warning(
                    f"Invalid format for error_code_search: '{error_code_search}'. Expected an integer. Skipping error code filter."
                )

        sort_column = getattr(ErrorLog, sort_by, ErrorLog.id)
        if sort_order.lower() == "asc":
            query = query.order_by(asc(sort_column))
        else:
            query = query.order_by(desc(sort_column))

        query = query.limit(limit).offset(offset)

        result = await session.execute(query)
        rows = result.fetchall()
        return [dict(row._mapping) for row in rows]
    except Exception as e:
        logger.error("Failed to get error logs with filter: %s", e, exc_info=True)
        raise


async def get_error_logs_count(
    session: AsyncSession,
    key_search: Optional[str] = None,
    error_search: Optional[str] = None,
    error_code_search: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> int:
    """
    Get the total number of error logs that meet the conditions

    Args:
        key_search (Optional[str]): Gemini key search term (fuzzy match)
        error_search (Optional[str]): Error type or log content search term (fuzzy match)
        error_code_search (Optional[str]): Error code search term (exact match)
        start_date (Optional[datetime]): Start date and time
        end_date (Optional[datetime]): End date and time

    Returns:
        int: Total number of logs
    """
    try:
        query = select(func.count()).select_from(ErrorLog)

        if key_search:
            query = query.where(ErrorLog.gemini_key.ilike(f"%{key_search}%"))
        if error_search:
            query = query.where(
                (ErrorLog.error_type.ilike(f"%{error_search}%"))
                | (ErrorLog.error_log.ilike(f"%{error_search}%"))
            )
        if start_date:
            query = query.where(ErrorLog.request_time >= start_date)
        if end_date:
            query = query.where(ErrorLog.request_time < end_date)
        if error_code_search:
            try:
                error_code_int = int(error_code_search)
                query = query.where(ErrorLog.error_code == error_code_int)
            except ValueError:
                logger.warning(
                    f"Invalid format for error_code_search in count: '{error_code_search}'. Expected an integer. Skipping error code filter."
                )

        result = await session.execute(query)
        count = result.scalar_one()
        return count if count else 0
    except Exception as e:
        logger.error("Failed to count error logs with filters: %s", e, exc_info=True)
        raise


# New function: Get the details of a single error log
async def get_error_log_details(
    session: AsyncSession, log_id: int
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information of a single error log by ID

    Args:
        log_id (int): The ID of the error log

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the detailed information of the log, or None if not found
    """
    try:
        query = select(ErrorLog).where(ErrorLog.id == log_id)
        result = await session.execute(query)
        row = result.scalar_one_or_none()
        if row:
            # Convert request_msg (JSONB) to a string for return in the API
            log_dict = dict(row.__dict__)
            if "request_msg" in log_dict and log_dict["request_msg"] is not None:
                # Ensure that even None or non-JSON data can be handled
                try:
                    log_dict["request_msg"] = json.dumps(
                        log_dict["request_msg"], ensure_ascii=False, indent=2
                    )
                except TypeError:
                    log_dict["request_msg"] = str(log_dict["request_msg"])
            return log_dict
        else:
            return None
    except Exception as e:
        logger.error(
            "Failed to get error log details for ID %s: %s",
            log_id,
            e,
            exc_info=True,
        )
        raise


# New function: Find the closest error log by gemini_key / error_code / time window
async def find_error_log_by_info(
    session: AsyncSession,
    gemini_key: str,
    timestamp: datetime,
    status_code: Optional[int] = None,
    window_seconds: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Find the error log closest to the timestamp within a given time window, based on gemini_key (exact match) and optional status_code.

    It is assumed that the error_code of the error log stores the HTTP status code or an equivalent error code.

    Args:
        gemini_key: The complete Gemini key string.
        timestamp: The target time (UTC or local, consistent with storage).
        status_code: Optional error code, which will be matched first if provided.
        window_seconds: The allowed time deviation window in seconds, default is 1 second.

    Returns:
        Optional[Dict[str, Any]]: The complete details of the best matching error log (fields are consistent with get_error_log_details), or None if not found.
    """
    try:
        start_time = timestamp - timedelta(seconds=window_seconds)
        end_time = timestamp + timedelta(seconds=window_seconds)

        base_query = select(ErrorLog).where(
            ErrorLog.gemini_key == gemini_key,
            ErrorLog.request_time >= start_time,
            ErrorLog.request_time <= end_time,
        )

        # If a status code is provided, try to filter by status code first
        if status_code is not None:
            query = base_query.where(ErrorLog.error_code == status_code).order_by(
                ErrorLog.request_time.desc()
            )
            result = await session.execute(query)
            candidates = result.scalars().all()
            if not candidates:
                # Fallback: without status code, only by time window
                query2 = base_query.order_by(ErrorLog.request_time.desc())
                result2 = await session.execute(query2)
                candidates = result2.scalars().all()
        else:
            query = base_query.order_by(ErrorLog.request_time.desc())
            result = await session.execute(query)
            candidates = result.scalars().all()

        if not candidates:
            return None

        # Select the one closest to the timestamp in Python
        def _to_dict(row: Any) -> Dict[str, Any]:
            d = dict(row.__dict__)
            if "request_msg" in d and d["request_msg"] is not None:
                try:
                    d["request_msg"] = json.dumps(
                        d["request_msg"], ensure_ascii=False, indent=2
                    )
                except TypeError:
                    d["request_msg"] = str(d["request_msg"])
            return d

        best = min(
            candidates,
            key=lambda r: abs((r.request_time - timestamp).total_seconds()),
        )
        return _to_dict(best)
    except Exception as e:
        logger.error(
            "Failed to find error log by info (key=***%s, code=%s, ts=%s, window=%ss): %s",
            gemini_key[-4:] if gemini_key else "",
            status_code,
            timestamp,
            window_seconds,
            e,
            exc_info=True,
        )
        raise


async def delete_error_logs_by_ids(session: AsyncSession, log_ids: List[int]) -> int:
    """
    Bulk delete error logs based on the provided list of IDs (asynchronous).

    Args:
        log_ids: A list of error log IDs to delete.

    Returns:
        int: The number of logs actually deleted.

    NOTE:
        This function does not commit; callers using the FastAPI dependency should rely
        on the shared session's lifecycle (e.g., `get_db`) to handle commit/rollback.
    """
    if not log_ids:
        return 0
    try:
        # Perform the deletion; caller (get_db dependency) will handle commit/rollback
        query = delete(ErrorLog).where(ErrorLog.id.in_(log_ids))
        await session.execute(query)
        logger.info(f"Attempted bulk deletion for error logs with IDs: {log_ids}")
        return len(log_ids)  # Return the number of attempted deletions
    except Exception as e:
        # Database connection or execution error
        logger.error(
            f"Error during bulk deletion of error logs {log_ids}: {e}", exc_info=True
        )
        raise


async def delete_error_log_by_id(session: AsyncSession, log_id: int) -> bool:
    """
    Delete a single error log by ID (asynchronous).

    Args:
        log_id: The ID of the error log to delete.

    Returns:
        bool: Returns True if successfully deleted, otherwise returns False.
    """
    try:
        # Check for existence first (optional, but more explicit)
        check_query = select(ErrorLog.id).where(ErrorLog.id == log_id)
        result = await session.execute(check_query)
        exists = result.scalar_one_or_none()

        if not exists:
            logger.warning(
                f"Attempted to delete non-existent error log with ID: {log_id}"
            )
            return False

        # Perform the deletion
        delete_query = delete(ErrorLog).where(ErrorLog.id == log_id)
        await session.execute(delete_query)
        await session.commit()
        logger.info(f"Successfully deleted error log with ID: {log_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting error log with ID {log_id}: {e}", exc_info=True)
        raise


async def delete_all_error_logs(session: AsyncSession) -> int:
    """
    Delete all error logs in batches to avoid timeouts and performance issues with large amounts of data.

    Returns:
        int: The total number of error logs deleted.
    """
    total_deleted_count = 0
    # SQLite has a limit on the number of SQL parameters (commonly 999), and too many parameters in the IN clause will cause an error
    # Use 500 uniformly to be compatible with SQLite/MySQL, and this value can be exposed in the configuration if necessary
    batch_size = 200

    try:
        while True:
            # 1) Read a batch of IDs to be deleted, selecting only the ID column to improve efficiency
            id_query = select(ErrorLog.id).order_by(ErrorLog.id).limit(batch_size)
            result = await session.execute(id_query)
            rows = result.fetchall()
            if not rows:
                break

            ids = [row[0] for row in rows]

            # 2) Bulk delete by ID
            delete_query = delete(ErrorLog).where(ErrorLog.id.in_(ids))
            await session.execute(delete_query)
            await session.commit()

            deleted_in_batch = len(ids)
            total_deleted_count += deleted_in_batch

            logger.debug(f"Deleted a batch of {deleted_in_batch} error logs.")

            # If it is less than one batch, it means the deletion is complete
            if deleted_in_batch < batch_size:
                break

            # 3) Give control back to the event loop to alleviate long-term occupation
            await asyncio.sleep(0)

        logger.info(
            f"Successfully deleted all error logs in batches. Total deleted: {total_deleted_count}"
        )
        return total_deleted_count
    except Exception as e:
        logger.error(
            f"Failed to delete all error logs in batches: {str(e)}", exc_info=True
        )
        raise


# New function: Add request log
async def add_request_log(
    session: AsyncSession,
    model_name: Optional[str],
    api_key: Optional[str],
    is_success: bool,
    status_code: Optional[int] = None,
    latency_ms: Optional[int] = None,
    request_time: Optional[datetime] = None,
    token_count: Optional[int] = None,
) -> bool:
    """
    Add API request log

    Args:
        model_name: Model name
        api_key: API key used
        is_success: Whether the request was successful
        status_code: API response status code
        latency_ms: Request latency (milliseconds)
        request_time: Time of request (if None, current time is used)
        token_count: Token count for the request
    """
    try:
        log_time = request_time if request_time else datetime.now()

        query = insert(RequestLog).values(
            request_time=log_time,
            model_name=model_name,
            api_key=api_key,
            is_success=is_success,
            status_code=status_code,
            latency_ms=latency_ms,
            token_count=token_count,
        )
        await session.execute(query)
        await session.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to add request log: {str(e)}", exc_info=True)
        return False


# ==================== File record related functions ====================


async def create_file_record(
    session: AsyncSession,
    name: str,
    mime_type: str,
    size_bytes: int,
    api_key: str,
    uri: str,
    create_time: datetime,
    update_time: datetime,
    expiration_time: datetime,
    state: FileState = FileState.PROCESSING,
    display_name: Optional[str] = None,
    sha256_hash: Optional[str] = None,
    upload_url: Optional[str] = None,
    user_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create file record

    Args:
        name: File name (format: files/{file_id})
        mime_type: MIME type
        size_bytes: File size (bytes)
        api_key: API Key used for upload
        uri: File access URI
        create_time: Creation time
        update_time: Update time
        expiration_time: Expiration time
        display_name: Display name
        sha256_hash: SHA256 hash value
        upload_url: Temporary upload URL
        user_token: Token of the uploading user

    Returns:
        Dict[str, Any]: The created file record
    """
    try:
        query = insert(FileRecord).values(
            name=name,
            display_name=display_name,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256_hash=sha256_hash,
            state=state,
            create_time=create_time,
            update_time=update_time,
            expiration_time=expiration_time,
            uri=uri,
            api_key=api_key,
            upload_url=upload_url,
            user_token=user_token,
        )
        await session.execute(query)
        await session.commit()

        # Return the created record
        record = await get_file_record_by_name(session, name)
        if record is None:
            raise Exception(f"Failed to create or find file record: {name}")
        return record
    except Exception as e:
        logger.error(f"Failed to create file record: {str(e)}", exc_info=True)
        raise


async def get_file_record_by_name(
    session: AsyncSession, name: str
) -> Optional[Dict[str, Any]]:
    """
    Get file record by file name

    Args:
        name: File name (format: files/{file_id})

    Returns:
        Optional[Dict[str, Any]]: The file record, or None if it does not exist
    """
    try:
        query = select(FileRecord).where(FileRecord.name == name)
        result = await session.execute(query)
        row = result.scalar_one_or_none()
        return dict(row.__dict__) if row else None
    except Exception as e:
        logger.error(
            f"Failed to get file record by name {name}: {str(e)}", exc_info=True
        )
        raise


async def update_file_record_state(
    session: AsyncSession,
    file_name: str,
    state: FileState,
    update_time: Optional[datetime] = None,
    upload_completed: Optional[datetime] = None,
    sha256_hash: Optional[str] = None,
) -> bool:
    """
    Update file record state

    Args:
        file_name: File name
        state: New state
        update_time: Update time
        upload_completed: Upload completion time
        sha256_hash: SHA256 hash value

    Returns:
        bool: Whether the update was successful
    """
    try:
        values: Dict[str, Any] = {"state": state}
        if update_time:
            values["update_time"] = update_time
        if upload_completed:
            values["upload_completed"] = upload_completed
        if sha256_hash:
            values["sha256_hash"] = sha256_hash

        query = update(FileRecord).where(FileRecord.name == file_name).values(**values)
        await session.execute(query)
        await session.commit()

        # Check if update was successful by querying the record
        updated_record = await get_file_record_by_name(session, file_name)
        if updated_record:
            logger.info(f"Updated file record state for {file_name} to {state}")
            return True

        logger.warning(f"File record not found for update: {file_name}")
        return False
    except Exception as e:
        logger.error(f"Failed to update file record state: {str(e)}", exc_info=True)
        return False


async def list_file_records(
    session: AsyncSession,
    user_token: Optional[str] = None,
    api_key: Optional[str] = None,
    page_size: int = 10,
    page_token: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """
    List file records

    Args:
        user_token: User token (if provided, only files for that user are returned)
        api_key: API Key (if provided, only files using that key are returned)
        page_size: Page size
        page_token: Page token (offset)

    Returns:
        tuple[List[Dict[str, Any]], Optional[str]]: (List of files, next page token)
    """
    try:
        logger.debug(
            f"list_file_records called with page_size={page_size}, page_token={page_token}"
        )
        query = select(FileRecord).where(
            FileRecord.expiration_time > datetime.now(timezone.utc)
        )

        if user_token:
            query = query.where(FileRecord.user_token == user_token)
        if api_key:
            query = query.where(FileRecord.api_key == api_key)

        # Paginate using offset
        offset = 0
        if page_token:
            try:
                offset = int(page_token)
            except ValueError:
                logger.warning(f"Invalid page token: {page_token}")
                offset = 0

        # Sort by ID in ascending order, using OFFSET and LIMIT
        query = query.order_by(FileRecord.id).offset(offset).limit(page_size + 1)

        result = await session.execute(query)
        rows = result.scalars().all()

        logger.debug(f"Query returned {len(rows)} records")
        if rows:
            logger.debug(
                f"First record ID: {rows[0].id}, Last record ID: {rows[-1].id}"
            )

        # Handle pagination
        has_next = len(rows) > page_size
        if has_next:
            rows = rows[:page_size]
            # The offset for the next page is the current offset plus the number of records returned on this page
            next_offset = offset + page_size
            next_page_token = str(next_offset)
            logger.debug(
                f"Has next page, offset={offset}, page_size={page_size}, next_page_token={next_page_token}"
            )
        else:
            next_page_token = None
            logger.debug(f"No next page, returning {len(rows)} results")

        return [dict(row.__dict__) for row in rows], next_page_token
    except Exception as e:
        logger.error(f"Failed to list file records: {str(e)}", exc_info=True)
        raise


async def delete_file_record(session: AsyncSession, name: str) -> bool:
    """
    Delete file record

    Args:
        name: File name

    Returns:
        bool: Whether the deletion was successful
    """
    try:
        query = delete(FileRecord).where(FileRecord.name == name)
        await session.execute(query)
        await session.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to delete file record: {str(e)}", exc_info=True)
        return False


async def delete_expired_file_records(session: AsyncSession) -> List[Dict[str, Any]]:
    """
    Delete expired file records

    Returns:
        List[Dict[str, Any]]: List of deleted records
    """
    try:
        # First, get the records to be deleted
        query = select(FileRecord).where(
            FileRecord.expiration_time <= datetime.now(timezone.utc)
        )
        result = await session.execute(query)
        expired_records = result.scalars().all()

        if not expired_records:
            return []

        # Perform the deletion
        delete_query = delete(FileRecord).where(
            FileRecord.expiration_time <= datetime.now(timezone.utc)
        )
        await session.execute(delete_query)
        await session.commit()

        logger.info(f"Deleted {len(expired_records)} expired file records")
        return [dict(record.__dict__) for record in expired_records]
    except Exception as e:
        logger.error(f"Failed to delete expired file records: {str(e)}", exc_info=True)
        raise


async def get_file_api_key(session: AsyncSession, name: str) -> Optional[str]:
    """
    Get the API Key corresponding to the file

    Args:
        name: File name

    Returns:
        Optional[str]: The API Key, or None if the file does not exist or has expired
    """
    try:
        query = select(FileRecord.api_key).where(
            (FileRecord.name == name)
            & (FileRecord.expiration_time > datetime.now(timezone.utc))
        )
        result = await session.execute(query)
        row = result.first()
        return row[0] if row else None
    except Exception as e:
        logger.error(f"Failed to get file API key: {str(e)}", exc_info=True)
        raise
