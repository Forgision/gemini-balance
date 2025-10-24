"""
Database service module
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import asc, delete, desc, func, insert, select, update

from app.database.connection import database
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


async def get_all_settings() -> List[Dict[str, Any]]:
    """
    Get all settings

    Returns:
        List[Dict[str, Any]]: List of settings
    """
    try:
        query = select(Settings)
        result = await database.fetch_all(query)
        return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Failed to get all settings: {str(e)}")
        raise


async def get_setting(key: str) -> Optional[Dict[str, Any]]:
    """
    Get the setting for the specified key

    Args:
        key: Setting key name

    Returns:
        Optional[Dict[str, Any]]: Setting information, or None if it does not exist
    """
    try:
        query = select(Settings).where(Settings.key == key)
        result = await database.fetch_one(query)
        return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get setting {key}: {str(e)}")
        raise


async def update_setting(
    key: str, value: str, description: Optional[str] = None
) -> bool:
    """
    Update setting

    Args:
        key: Setting key name
        value: Setting value
        description: Setting description

    Returns:
        bool: Whether the update was successful
    """
    try:
        # Check if the setting exists
        setting = await get_setting(key)

        if setting:
            # Update setting
            query = (
                update(Settings)
                .where(Settings.key == key)
                .values(
                    value=value,
                    description=description if description else setting["description"],
                    updated_at=datetime.now(),
                )
            )
            await database.execute(query)
            logger.info(f"Updated setting: {key}")
            return True
        else:
            # Insert setting
            query = insert(Settings).values(
                key=key,
                value=value,
                description=description,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            await database.execute(query)
            logger.info(f"Inserted setting: {key}")
            return True
    except Exception as e:
        logger.error(f"Failed to update setting {key}: {str(e)}")
        return False


async def get_usage_stats_by_key_and_model(
    api_key: str, model_name: str
) -> Optional[Dict[str, Any]]:
    """
    Get usage statistics for a given API key and model.

    Args:
        api_key: The API key.
        model_name: The model name.

    Returns:
        Optional[Dict[str, Any]]: The usage statistics, or None if not found.
    """
    try:
        query = (
            select(UsageStats)
            .where(
                UsageStats.api_key == api_key,
                UsageStats.model_name == model_name,
            )
            .order_by(desc(UsageStats.timestamp))
        )
        result = await database.fetch_one(query)
        return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get usage stats: {str(e)}")
        raise


async def get_all_usage_stats() -> List[Dict[str, Any]]:
    """
    Get all usage statistics.

    Returns:
        List[Dict[str, Any]]: A list of usage statistics.
    """
    try:
        query = select(UsageStats)
        result = await database.fetch_all(query)
        return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Failed to get all usage stats: {str(e)}")
        raise


async def add_error_log(
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
        await database.execute(query)
        logger.info(f"Added error log for key: {redact_key_for_logging(gemini_key if gemini_key is not None else 'N/A')}")
        return True
    except Exception as e:
        logger.error(f"Failed to add error log: {str(e)}")
        return False


async def get_error_logs(
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

        result = await database.fetch_all(query)
        return [dict(row) for row in result]
    except Exception as e:
        logger.exception(f"Failed to get error logs with filters: {str(e)}")
        raise


async def get_error_logs_count(
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

        count_result = await database.fetch_one(query)
        return count_result[0] if count_result else 0
    except Exception as e:
        logger.exception(f"Failed to count error logs with filters: {str(e)}")
        raise


# New function: Get the details of a single error log
async def get_error_log_details(log_id: int) -> Optional[Dict[str, Any]]:
    """
    Get detailed information of a single error log by ID

    Args:
        log_id (int): The ID of the error log

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the detailed information of the log, or None if not found
    """
    try:
        query = select(ErrorLog).where(ErrorLog.id == log_id)
        result = await database.fetch_one(query)
        if result:
            # Convert request_msg (JSONB) to a string for return in the API
            log_dict = dict(result)
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
        logger.exception(f"Failed to get error log details for ID {log_id}: {str(e)}")
        raise


# New function: Find the closest error log by gemini_key / error_code / time window
async def find_error_log_by_info(
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
            candidates = await database.fetch_all(query)
            if not candidates:
                # Fallback: without status code, only by time window
                query2 = base_query.order_by(ErrorLog.request_time.desc())
                candidates = await database.fetch_all(query2)
        else:
            query = base_query.order_by(ErrorLog.request_time.desc())
            candidates = await database.fetch_all(query)

        if not candidates:
            return None

        # Select the one closest to the timestamp in Python
        def _to_dict(row: Any) -> Dict[str, Any]:
            d = dict(row)
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
            key=lambda r: abs((r["request_time"] - timestamp).total_seconds()),
        )
        return _to_dict(best)
    except Exception as e:
        logger.exception(
            f"Failed to find error log by info (key=***{gemini_key[-4:] if gemini_key else ''}, code={status_code}, ts={timestamp}, window={window_seconds}s): {str(e)}"
        )
        raise


async def delete_error_logs_by_ids(log_ids: List[int]) -> int:
    """
    Bulk delete error logs based on the provided list of IDs (asynchronous).

    Args:
        log_ids: A list of error log IDs to delete.

    Returns:
        int: The number of logs actually deleted.
    """
    if not log_ids:
        return 0
    try:
        # Use databases to perform the deletion
        query = delete(ErrorLog).where(ErrorLog.id.in_(log_ids))
        # execute returns the number of affected rows, but the databases library's execute does not directly return rowcount
        # We need to query for existence first, or rely on database constraints/triggers (if applicable)
        # Alternatively, we can perform the deletion and assume success unless an exception is thrown
        # For simplicity, we perform the deletion and log it, without accurately returning the number of deletions
        # If an accurate count is needed, a SELECT COUNT(*) needs to be executed first
        await database.execute(query)
        # Note: databases' execute does not return rowcount, so we cannot directly return the number of deletions
        # Return the length of log_ids as the number of attempted deletions, or return 0/1 to indicate that the operation was attempted
        logger.info(f"Attempted bulk deletion for error logs with IDs: {log_ids}")
        return len(log_ids)  # Return the number of attempted deletions
    except Exception as e:
        # Database connection or execution error
        logger.error(
            f"Error during bulk deletion of error logs {log_ids}: {e}", exc_info=True
        )
        raise


async def delete_error_log_by_id(log_id: int) -> bool:
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
        exists = await database.fetch_one(check_query)

        if not exists:
            logger.warning(
                f"Attempted to delete non-existent error log with ID: {log_id}"
            )
            return False

        # Perform the deletion
        delete_query = delete(ErrorLog).where(ErrorLog.id == log_id)
        await database.execute(delete_query)
        logger.info(f"Successfully deleted error log with ID: {log_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting error log with ID {log_id}: {e}", exc_info=True)
        raise


async def delete_all_error_logs() -> int:
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
            rows = await database.fetch_all(id_query)
            if not rows:
                break

            ids = [row["id"] for row in rows]

            # 2) Bulk delete by ID
            delete_query = delete(ErrorLog).where(ErrorLog.id.in_(ids))
            await database.execute(delete_query)

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
    model_name: Optional[str],
    api_key: Optional[str],
    is_success: bool,
    status_code: Optional[int] = None,
    latency_ms: Optional[int] = None,
    request_time: Optional[datetime] = None,
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

    Returns:
        bool: Whether the addition was successful
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
        )
        await database.execute(query)
        return True
    except Exception as e:
        logger.error(f"Failed to add request log: {str(e)}")
        return False


# ==================== File record related functions ====================


async def create_file_record(
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
        await database.execute(query)

        # Return the created record
        record = await get_file_record_by_name(name)
        if record is None:
            raise Exception(f"Failed to create or find file record: {name}")
        return record
    except Exception as e:
        logger.error(f"Failed to create file record: {str(e)}")
        raise


async def get_file_record_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Get file record by file name

    Args:
        name: File name (format: files/{file_id})

    Returns:
        Optional[Dict[str, Any]]: The file record, or None if it does not exist
    """
    try:
        query = select(FileRecord).where(FileRecord.name == name)
        result = await database.fetch_one(query)
        return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get file record by name {name}: {str(e)}")
        raise


async def update_file_record_state(
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
        result = await database.execute(query)

        if result:
            logger.info(f"Updated file record state for {file_name} to {state}")
            return True

        logger.warning(f"File record not found for update: {file_name}")
        return False
    except Exception as e:
        logger.error(f"Failed to update file record state: {str(e)}")
        return False


async def list_file_records(
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

        results = await database.fetch_all(query)

        logger.debug(f"Query returned {len(results)} records")
        if results:
            logger.debug(
                f"First record ID: {results[0]['id']}, Last record ID: {results[-1]['id']}"
            )

        # Handle pagination
        has_next = len(results) > page_size
        if has_next:
            results = results[:page_size]
            # The offset for the next page is the current offset plus the number of records returned on this page
            next_offset = offset + page_size
            next_page_token = str(next_offset)
            logger.debug(
                f"Has next page, offset={offset}, page_size={page_size}, next_page_token={next_page_token}"
            )
        else:
            next_page_token = None
            logger.debug(f"No next page, returning {len(results)} results")

        return [dict(row) for row in results], next_page_token
    except Exception as e:
        logger.error(f"Failed to list file records: {str(e)}")
        raise


async def delete_file_record(name: str) -> bool:
    """
    Delete file record

    Args:
        name: File name

    Returns:
        bool: Whether the deletion was successful
    """
    try:
        query = delete(FileRecord).where(FileRecord.name == name)
        await database.execute(query)
        return True
    except Exception as e:
        logger.error(f"Failed to delete file record: {str(e)}")
        return False


async def delete_expired_file_records() -> List[Dict[str, Any]]:
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
        expired_records = await database.fetch_all(query)

        if not expired_records:
            return []

        # Perform the deletion
        delete_query = delete(FileRecord).where(
            FileRecord.expiration_time <= datetime.now(timezone.utc)
        )
        await database.execute(delete_query)

        logger.info(f"Deleted {len(expired_records)} expired file records")
        return [dict(record) for record in expired_records]
    except Exception as e:
        logger.error(f"Failed to delete expired file records: {str(e)}")
        raise


async def get_file_api_key(name: str) -> Optional[str]:
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
        result = await database.fetch_one(query)
        return result["api_key"] if result else None
    except Exception as e:
        logger.error(f"Failed to get file API key: {str(e)}")
        raise


async def update_usage_stats(api_key: str, model_name: str, token_count: int) -> bool:
    """
    Update usage statistics.

    Args:
        api_key: The API key used.
        model_name: The model name used.
        token_count: The number of tokens used.

    Returns:
        bool: Whether the update was successful.
    """
    try:
        now = datetime.now()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        async with database.transaction():
            # Find the record for today
            query = select(UsageStats).where(
                UsageStats.api_key == api_key,
                UsageStats.model_name == model_name,
                UsageStats.timestamp >= start_of_day,
            )
            record = await database.fetch_one(query)

            if record:
                # Update existing record
                update_query = (
                    update(UsageStats)
                    .where(UsageStats.id == record["id"])
                    .values(
                        rpm=UsageStats.rpm + 1,
                        rpd=UsageStats.rpd + 1,
                        token_count=UsageStats.token_count + token_count,
                        timestamp=now,
                    )
                )
                await database.execute(update_query)
            else:
                # Insert new record
                insert_query = insert(UsageStats).values(
                    api_key=api_key,
                    model_name=model_name,
                    rpm=1,
                    rpd=1,
                    token_count=token_count,
                    timestamp=now,
                )
                await database.execute(insert_query)

        return True
    except Exception as e:
        logger.error(f"Failed to update usage stats: {str(e)}")
        return False
