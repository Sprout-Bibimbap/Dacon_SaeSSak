import time
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware


class StartTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request.state.start_time = start_time
        response = await call_next(request)
        return response
