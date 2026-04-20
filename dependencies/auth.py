from fastapi import Header
from core.config import settings
from core.exceptions import ServiceAuthError


async def verify_internal_token(x_internal_token: str = Header(None)) -> None:
    if x_internal_token != settings.INTERNAL_TOKEN:
        raise ServiceAuthError("Invalid or missing internal token")
