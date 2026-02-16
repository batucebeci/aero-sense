from __future__ import annotations

import os

from fastapi import FastAPI, Request, status
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

PROTECTED_METHODS = {"POST", "PUT", "DELETE", "PATCH"}
PUBLIC_PATH_PREFIXES = ("/static", "/docs", "/openapi.json", "/redoc", "/metrics", "/api/health")


def api_key() -> str | None:
    return os.environ.get("API_KEY")


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        key = api_key()
        if key:
            path = request.url.path
            if not any(path.startswith(p) for p in PUBLIC_PATH_PREFIXES) and request.method in PROTECTED_METHODS:
                header_key = request.headers.get("x-api-key")
                query_key = request.query_params.get("api_key")
                if header_key != key and query_key != key:
                    return JSONResponse(
                        {"detail": "invalid or missing api key"},
                        status_code=status.HTTP_401_UNAUTHORIZED,
                    )
        return await call_next(request)


def install(app: FastAPI) -> None:
    limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(APIKeyMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": "rate limit exceeded", "retry_after": getattr(exc, "retry_after", None)},
        )
