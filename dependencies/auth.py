from fastapi import Header, HTTPException, status
from core.config import settings


async def verify_internal_token(x_internal_token: str = Header(None)):
    """Verify that the X-Internal-Token header matches the configured token."""
    if not settings.INTERNAL_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal token not configured",
        )

    if x_internal_token is None or x_internal_token != settings.INTERNAL_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal token",
        )
