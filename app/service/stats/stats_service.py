# app/service/stats_service.py

import datetime
from typing import Union

from sqlalchemy import and_, case, func, or_, select

from app.database.connection import database
from app.database.models import RequestLog
from app.log.logger import get_stats_logger

logger = get_stats_logger()


class StatsService:
    """Service class for handling statistics related operations."""

    async def get_calls_in_last_seconds(self, seconds: int) -> dict[str, int]:
        """Get the number of calls in the last N seconds (total, success, failure)."""
        try:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(seconds=seconds)
            query = select(
                func.count(RequestLog.id).label("total"),
                func.sum(
                    case(
                        (
                            and_(
                                RequestLog.status_code >= 200,
                                RequestLog.status_code < 300,
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ).label("success"),
                func.sum(
                    case(
                        (
                            (
                                or_(
                                    RequestLog.status_code < 200,
                                    RequestLog.status_code >= 300,
                                ),
                                1,
                            ),
                            (RequestLog.status_code.is_(None), 1),
                        ),
                        else_=0,
                    )
                ).label("failure"),
            ).where(RequestLog.request_time >= cutoff_time)
            result = await database.fetch_one(query)
            if result:
                return {
                    "total": result["total"] or 0,
                    "success": result["success"] or 0,
                    "failure": result["failure"] or 0,
                }
            return {"total": 0, "success": 0, "failure": 0}
        except Exception as e:
            logger.error(f"Failed to get calls in last {seconds} seconds: {e}")
            return {"total": 0, "success": 0, "failure": 0}

    async def get_calls_in_last_minutes(self, minutes: int) -> dict[str, int]:
        """Get the number of calls in the last N minutes (total, success, failure)."""
        return await self.get_calls_in_last_seconds(minutes * 60)

    async def get_calls_in_last_hours(self, hours: int) -> dict[str, int]:
        """Get the number of calls in the last N hours (total, success, failure)."""
        return await self.get_calls_in_last_seconds(hours * 3600)

    async def get_calls_in_current_month(self) -> dict[str, int]:
        """Get the number of calls in the current calendar month (total, success, failure)."""
        try:
            now = datetime.datetime.now()
            start_of_month = now.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            query = select(
                func.count(RequestLog.id).label("total"),
                func.sum(
                    case(
                        (
                            and_(
                                RequestLog.status_code >= 200,
                                RequestLog.status_code < 300,
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ).label("success"),
                func.sum(
                    case(
                        (
                            (
                                or_(
                                    RequestLog.status_code < 200,
                                    RequestLog.status_code >= 300,
                                ),
                                1,
                            ),
                            (RequestLog.status_code.is_(None), 1),
                        ),
                        else_=0,
                    )
                ).label("failure"),
            ).where(RequestLog.request_time >= start_of_month)
            result = await database.fetch_one(query)
            if result:
                return {
                    "total": result["total"] or 0,
                    "success": result["success"] or 0,
                    "failure": result["failure"] or 0,
                }
            return {"total": 0, "success": 0, "failure": 0}
        except Exception as e:
            logger.error(f"Failed to get calls in current month: {e}")
            return {"total": 0, "success": 0, "failure": 0}

    async def get_api_usage_stats(self) -> dict:
        """Get all required API usage statistics (total, success, failure)."""
        try:
            stats_1m = await self.get_calls_in_last_minutes(1)
            stats_1h = await self.get_calls_in_last_hours(1)
            stats_24h = await self.get_calls_in_last_hours(24)
            stats_month = await self.get_calls_in_current_month()

            return {
                "calls_1m": stats_1m,
                "calls_1h": stats_1h,
                "calls_24h": stats_24h,
                "calls_month": stats_month,
            }
        except Exception as e:
            logger.error(f"Failed to get API usage stats: {e}")
            default_stat = {"total": 0, "success": 0, "failure": 0}
            return {
                "calls_1m": default_stat.copy(),
                "calls_1h": default_stat.copy(),
                "calls_24h": default_stat.copy(),
                "calls_month": default_stat.copy(),
            }

    async def get_api_call_details(self, period: str) -> list[dict]:
        """
        Get API call details for the specified period.

        Args:
            period: Time period identifier ('1m', '1h', '24h')

        Returns:
            A list of dictionaries containing call details, each dictionary includes timestamp, key, model, status, status_code, latency_ms, error_log_id (optional).

        Raises:
            ValueError: If the period is invalid.
        """
        now = datetime.datetime.now()
        if period == "1m":
            start_time = now - datetime.timedelta(minutes=1)
        elif period == "1h":
            start_time = now - datetime.timedelta(hours=1)
        elif period == "8h":
            start_time = now - datetime.timedelta(hours=8)
        elif period == "24h":
            start_time = now - datetime.timedelta(hours=24)
        else:
            raise ValueError(f"Invalid time period identifier: {period}")

        try:
            query = (
                select(
                    RequestLog.request_time.label("timestamp"),
                    RequestLog.api_key.label("key"),
                    RequestLog.model_name.label("model"),
                    RequestLog.status_code.label("status_code"),
                    RequestLog.latency_ms.label("latency_ms"),
                )
                .where(RequestLog.request_time >= start_time)
                .order_by(RequestLog.request_time.desc())
            )

            results = await database.fetch_all(query)

            details: list[dict] = []
            for row in results:
                status = "failure"
                if row["status_code"] is not None:
                    status = "success" if 200 <= row["status_code"] < 300 else "failure"

                record = {
                    "timestamp": row["timestamp"].isoformat(),
                    "key": row["key"],
                    "model": row["model"],
                    "status": status,
                    "status_code": row["status_code"],
                    "latency_ms": row["latency_ms"],
                }

                details.append(record)

            logger.info(
                f"Retrieved {len(details)} API call details for period '{period}'"
            )
            return details

        except Exception as e:
            logger.error(f"Failed to get API call details for period '{period}': {e}")
            raise

    async def get_key_call_details(self, key: str, period: str) -> list[dict]:
        """Get call details for the specified key and period (same structure as get_api_call_details)."""
        now = datetime.datetime.now()
        if period == "1m":
            start_time = now - datetime.timedelta(minutes=1)
        elif period == "1h":
            start_time = now - datetime.timedelta(hours=1)
        elif period == "8h":
            start_time = now - datetime.timedelta(hours=8)
        elif period == "24h":
            start_time = now - datetime.timedelta(hours=24)
        else:
            raise ValueError(f"Invalid time period identifier: {period}")

        try:
            query = (
                select(
                    RequestLog.request_time.label("timestamp"),
                    RequestLog.api_key.label("key"),
                    RequestLog.model_name.label("model"),
                    RequestLog.status_code.label("status_code"),
                    RequestLog.latency_ms.label("latency_ms"),
                )
                .where(RequestLog.request_time >= start_time, RequestLog.api_key == key)
                .order_by(RequestLog.request_time.desc())
            )

            results = await database.fetch_all(query)

            details: list[dict] = []
            for row in results:
                status = "failure"
                if row["status_code"] is not None:
                    status = "success" if 200 <= row["status_code"] < 300 else "failure"

                record = {
                    "timestamp": row["timestamp"].isoformat(),
                    "key": row["key"],
                    "model": row["model"],
                    "status": status,
                    "status_code": row["status_code"],
                    "latency_ms": row["latency_ms"],
                }

                details.append(record)

            logger.info(
                f"Retrieved {len(details)} key call details for key=...{key[-4:] if key else ''} period '{period}'"
            )
            return details
        except Exception as e:
            logger.error(
                f"Failed to get key call details for key=...{key[-4:] if key else ''} period '{period}': {e}"
            )
            raise

    async def get_attention_keys_last_24h(
        self, include_keys: set[str], limit: int = 20, status_code: int = 429
    ) -> list[dict]:
        """Returns the list of keys with the most specified status codes (default 429) in the last 24 hours, including only the keys in include_keys.

        Returns: [{"key": str, "count": int, "status_code": int}, ...] sorted by count in descending order.
        """
        try:
            now = datetime.datetime.now()
            start_time = now - datetime.timedelta(hours=24)
            if not include_keys:
                return []
            query = (
                select(
                    RequestLog.api_key.label("key"),
                    func.count(RequestLog.id).label("count"),
                )
                .where(
                    RequestLog.request_time >= start_time,
                    RequestLog.status_code == status_code,
                    RequestLog.api_key.isnot(None),
                    RequestLog.api_key.in_(list(include_keys)),
                )
                .group_by(RequestLog.api_key)
                .order_by(func.count(RequestLog.id).desc())
                .limit(limit)
            )
            rows = await database.fetch_all(query)
            return [
                {"key": row["key"], "count": row["count"], "status_code": status_code}
                for row in rows
                if row["key"]
            ]
        except Exception as e:
            logger.error(
                f"Failed to get attention keys ({status_code}) in last 24h: {e}"
            )
            return []

    async def get_key_usage_details_last_24h(self, key: str) -> Union[dict, None]:
        """
        Get the number of calls for the specified API key in the last 24 hours, grouped by model.

        Args:
            key: The API key to query.

        Returns:
            A dictionary where the keys are model names and the values are the number of calls.
            May return None or an empty dictionary if an error occurs or no records are found.
            Example: {"gemini-pro": 10, "gemini-1.5-pro-latest": 5}
        """
        logger.info(
            f"Fetching usage details for key ending in ...{key[-4:]} for the last 24h."
        )
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=24)

        try:
            query = (
                select(
                    RequestLog.model_name, func.count(RequestLog.id).label("call_count")
                )
                .where(
                    RequestLog.api_key == key,
                    RequestLog.request_time >= cutoff_time,
                    RequestLog.model_name.isnot(None),
                )
                .group_by(RequestLog.model_name)
                .order_by(func.count(RequestLog.id).desc())
            )

            results = await database.fetch_all(query)

            if not results:
                logger.info(
                    f"No usage details found for key ending in ...{key[-4:]} in the last 24h."
                )
                return {}

            usage_details = {row["model_name"]: row["call_count"] for row in results}
            logger.info(
                f"Successfully fetched usage details for key ending in ...{key[-4:]}: {usage_details}"
            )
            return usage_details

        except Exception as e:
            logger.error(
                f"Failed to get key usage details for key ending in ...{key[-4:]}: {e}",
                exc_info=True,
            )
            raise
