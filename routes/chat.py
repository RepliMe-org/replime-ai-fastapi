from fastapi import APIRouter, Depends

from core.dependencies import verify_internal_token
from schemas.chat import ChatProcessRequest, ChatProcessResponse
from services.chat_service import process_chat

router = APIRouter(dependencies=[Depends(verify_internal_token)])


@router.post("/chat/process", response_model=ChatProcessResponse)
async def chat_process(request: ChatProcessRequest) -> ChatProcessResponse:
    return await process_chat(request)
