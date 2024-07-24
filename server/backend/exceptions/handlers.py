from fastapi import Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from exceptions.loggers import ControllerLogger
import logging
import json

level_ERROR = logging.ERROR
level_INFO = logging.INFO


async def get_data(request: Request):
    body = await request.body()
    if body:
        data = json.loads(body)
    else:
        data = {}
    return data


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    background_tasks = BackgroundTasks()
    data = await get_data(request)
    await ControllerLogger.logging(
        level_ERROR,
        "API request body에 문제가 있습니다.",
        exc,
        request,
        400,
        data,
        background_tasks,
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": "INVALID_PARAM",
            "message": "API request body에 문제가 있습니다.",
        },
        background=background_tasks,
    )


# API 오류
async def http_exception_handler(request: Request, exc: HTTPException):
    background_tasks = BackgroundTasks()
    data = await get_data(request)
    if exc.status_code == 400:
        await ControllerLogger.logging(
            level_ERROR,
            exc.detail,
            exc,
            request,
            exc.status_code,
            data,
            background_tasks,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "BAD_REQUEST", "message": exc.detail},
            background=background_tasks,
        )
    if exc.status_code == 401:
        await ControllerLogger.logging(
            level_ERROR,
            exc.detail,
            exc,
            request,
            exc.status_code,
            data,
            background_tasks,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "UNAUTHORIZED", "message": exc.detail},
            background=background_tasks,
        )
    elif exc.status_code == 404:
        await ControllerLogger.logging(
            level_ERROR,
            exc.detail,
            exc,
            request,
            exc.status_code,
            data,
            background_tasks,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "NO_DATA", "message": exc.detail},
            background=background_tasks,
        )
    elif exc.status_code == 409:
        await ControllerLogger.logging(
            level_ERROR,
            exc.detail,
            exc,
            request,
            exc.status_code,
            data,
            background_tasks,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "ALREADY_PROCESSED", "message": exc.detail},
            background=background_tasks,
        )
    elif exc.status_code == 422:
        await ControllerLogger.logging(
            level_ERROR,
            exc.detail,
            exc,
            request,
            exc.status_code,
            data,
            background_tasks,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "UNPROCESSABLE_ENTITY", "message": exc.detail},
            background=background_tasks,
        )
    elif exc.status_code == 503:
        await ControllerLogger.logging(
            level_ERROR,
            exc.detail,
            exc,
            request,
            exc.status_code,
            data,
            background_tasks,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "SERVICE_UNAVAILABLE", "message": exc.detail},
            background=background_tasks,
        )
    else:
        await ControllerLogger.logging(
            level_ERROR,
            exc.detail,
            exc,
            request,
            exc.status_code,
            data,
            background_tasks,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "ETC Error", "message": exc.detail},
            background=background_tasks,
        )


async def generic_exception_handler(request: Request, exc: Exception):
    background_tasks = BackgroundTasks()
    data = await get_data(request)
    error_type = exc.__class__.__name__
    error_message = str(exc)
    if exc.args:
        error_message = " ".join(str(arg) for arg in exc.args)
    await ControllerLogger.logging(
        level_ERROR, "Global error", exc, request, 500, data, background_tasks
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Global Error", "detail": f"{error_type} : {error_message}"},
        background=background_tasks,
    )
