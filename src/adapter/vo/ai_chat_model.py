from typing import List, Optional, Any
from pydantic import BaseModel


class MessageVO(BaseModel):
    message_id: str = None
    content: str
    role: str = None
    custom: str = None


class ChatInputVO(BaseModel):
    conversation_id: str | None = None
    messages: List[MessageVO]
    # input: str | None = None


class AiChatResultVO(BaseModel):
    message_id: str = None          # 消息id
    text: Optional[str] = None
    type: str = 'stream'    # stream | block
    role: str = 'assistant' # assistant | dataset_agent | code_agent | adapter_agent
