from fastapi import APIRouter
from sse_starlette import EventSourceResponse, ServerSentEvent

from src.adapter.vo.ai_chat_model import ChatInputVO
from src.app.ai_chat_service import chat_handler

from src.utils.SnowFlake import Snowflake

router = APIRouter(
    prefix="/aichat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)
sf = Snowflake(worker_id=0, datacenter_id=0)


@router.post("/chat", summary="会话列表")
async def chat(request: ChatInputVO) -> EventSourceResponse:
    return EventSourceResponse(
        content=chat_handler(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
        ping_message_factory=lambda: ServerSentEvent(
            event="ping", retry=15000, data=""
        ),
    )
