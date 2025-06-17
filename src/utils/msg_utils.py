import time

from langgraph.config import get_stream_writer

from src.adapter.vo.ai_chat_model import AiChatResultVO


class MessageBox(object):
    def __init__(self, type: str = 'stream', role: str = 'assistant', message_id: str = None):
        self.message_id = f"{role}-{time.time()}" if not message_id else message_id
        self.type = type
        self.role = role
        self.stream_writer = get_stream_writer()

    def new_box(self, type: str = 'stream', role: str = 'assistant'):
        self.message_id = f"{role}-{time.time()}"
        self.type = type
        self.role = role
        self.stream_writer = get_stream_writer()
        return self

    def write(self, text: str):
        # await self.flush()  # 每次写入后自动刷新
        self.stream_writer({"data": AiChatResultVO(
            message_id=self.message_id,
            text=text,
            type=self.type,
            role=self.role,
        ).model_dump_json(exclude_none=True)})
        return self

    async def flush(self):
        async for _ in self.async_gen():
            print("flush")


    # 异步生成器
    async def async_gen(self):
        import asyncio
        await asyncio.sleep(0.001)  # 模拟IO或延时
        yield

    @classmethod
    def write_block(cls, text: str, role: str='assistant'):
        return MessageBox(type='block', role=role).write(text)
        