from fastapi import APIRouter, Depends
from controllers import get_ai_controller, AIController
from dependencies import verify_internal_token

router = APIRouter()


@router.get("/health", dependencies=[Depends(verify_internal_token)])
def health_check(controller: AIController = Depends(get_ai_controller)):
    """
    Health check endpoint for AI service.
    Requires X-Internal-Token header.
    """
    return controller.health()
