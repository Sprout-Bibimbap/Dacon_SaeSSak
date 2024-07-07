import sys
import pytz
import time
import logging
import datetime
import traceback
from config import settings 
from typing import Any, Dict, Optional
from fastapi import Request, BackgroundTasks


class MongoLogger:
    mongo_client = None
    korea_timezone = pytz.timezone("Asia/Seoul")

    @classmethod
    async def initialize(cls, app) -> None:
        """Elasticsearch 클라이언트, teams bot, 로거를 초기화"""
        cls.mongo_client = app.state.mongo_client

    @classmethod
    async def log(
        cls,
        level: int,
        message: str,
        exception_info: Optional[Dict[str, Any]] = None,
        request_info: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
        response_time: Optional[float] = None,
        data_size: Optional[int] = None,
    ) -> None:
        """로그 정보를 Elasticsearch 및 파일에 기록"""

        korea_timezone = pytz.timezone("Asia/Seoul")
        utc_time = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        kst_time = utc_time.astimezone(korea_timezone).isoformat()
        log_entry = {
            "timestamp": kst_time,
            "level": level,
            "message": message,
            "request_info": request_info,
            "status_code": status_code,
            "response_time": response_time,
            "data_size": data_size,
        }
        if exception_info:
            log_entry["exception"] = exception_info

        index_name = (
            settings.ERROR_LOG_COLLECTION
            if level == logging.ERROR
            else settings.INFO_LOG_COLLECTION
        )
        try:
            await cls.mongo_client.index(index=index_name, document=log_entry)
        except Exception as e:
            error_message = f"Elasticsearch indexing failed: {str(e)}"
            print(error_message)

    @classmethod
    async def logging(
        cls,
        level: int,
        message: str,
        exc: Optional[Exception],
        request: Request,
        status_code: int,
        data: Optional[Dict[str, Any]] = None,
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> None:
        """정보 또는 오류 메시지를 로그"""
        cls.setup_logger()

        start_time = request.state.start_time
        end_time = time.time()
        response_time = (end_time - start_time) * 1000
        data_size = sys.getsizeof(data) if data else 0

        user_id_data = data.get("UserID", "Unknown") if data else "Unknown"

        user_id = getattr(request.state, "user_id", user_id_data)
        request_id = data.get("RequestID", "Unknown") if data else "Unknown"

        request_info = {
            "Api_name": request.scope["endpoint"].__doc__,
            "UserID": request.path_params.get("UserID", user_id),
            "RequestID": request.path_params.get("RequestID", request_id),
            "ClientIP": request.client.host,
            "UserAgent": request.headers.get("User-Agent", "Unknown"),
        }

        exc_info = None
        if exc and level == logging.ERROR:
            exc_info = {
                "type": type(exc).__name__,
                "description": str(exc),
                "traceback": traceback.format_exc(),
            }
        if background_tasks:
            background_tasks.add_task(
                cls.log,
                level,
                message,
                exc_info,
                request_info,
                status_code,
                response_time,
                data_size,
            )
        else:
            await cls.log(
                level,
                message,
                exc_info,
                request_info,
                status_code,
                response_time,
                data_size,
            )
