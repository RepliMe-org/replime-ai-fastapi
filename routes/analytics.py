from fastapi import APIRouter, Depends

from core.dependencies import verify_internal_token
from schemas.analytics import AnalyticsRequest, AnalyticsResponse
from services.analytics_service import compute_analytics

router = APIRouter(dependencies=[Depends(verify_internal_token)])


@router.post("/analytics/process", response_model=AnalyticsResponse)
async def analytics_process(request: AnalyticsRequest) -> AnalyticsResponse:
    return await compute_analytics(request)
