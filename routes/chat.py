from fastapi import APIRouter, BackgroundTasks, Depends

from core.dependencies import verify_internal_token
from schemas.chat import ChatProcessRequest, ChatProcessResponse
from services.chat_service import process_chat
from services.classification_service import classify_and_report

router = APIRouter(dependencies=[Depends(verify_internal_token)])


@router.post("/chat/process", response_model=ChatProcessResponse)
async def chat_process(
    request: ChatProcessRequest,
    background_tasks: BackgroundTasks,
) -> ChatProcessResponse:
    response = await process_chat(request)

    if request.message_classes:
        background_tasks.add_task(
            classify_and_report,
            request.message_id,
            request.query,
            request.message_classes,
        )

    return response
