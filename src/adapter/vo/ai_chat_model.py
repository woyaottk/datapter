from typing import List, Optional, Any
from pydantic import BaseModel


class MessageVO(BaseModel):
    content: str
    role: str = None
    custom: str = None


class ChatInputVO(BaseModel):
    conversation_id: str | None = None
    messages: List[MessageVO]
    # input: str | None = None


class AiChatResultVO(BaseModel):
    text: Optional[str] = None
    html: Optional[str] = None
    custom: Optional[Any] = None
