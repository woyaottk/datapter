import os

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langgraph.config import get_stream_writer
from langgraph.types import Command

from src.adapter.vo.ai_chat_model import AiChatResultVO
from src.domain.constant.constant import AgentTypeEnum
from src.domain.model.model import AdapterState, command_update
from src.utils.llm_util import async_create_llm


SYSTEM_PROMPT = """
# 角色
你是一个代码适配器智能机器人

# 场景
根据用户的问题进行回答
"""


class DemoAgent:
    def __init__(self):
        pass

    async def __call__(self, state: AdapterState) -> Command:
        conversation_id = state["conversationId"]
        print(f"[DemoAgent] called, state: {{'conversationId': {conversation_id}}}")
        print("这是Coordinator传递给我的context：" + state["context"])

        user_question = state["messages"][-1].content

        prompts = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
                ("user", "{user_question}"),
            ]
        )
        llm = await async_create_llm(
            **{
                "model_name": os.getenv("QWEN"),
                "api_key": os.getenv("ALIBABA_API_KEY"),
                "api_base": os.getenv("ALIBABA_BASE_URL"),
                "temperature": 0.2,
            }
        )
        chain = prompts | llm

        writer = get_stream_writer()
        state["context"] = ""
        async for chunk in chain.astream(
            {
                "user_question": user_question,
            }
        ):
            if chunk.content:
                state["context"] = state["context"] + chunk.content
                writer(
                    {
                        "data": AiChatResultVO(text=chunk.content).model_dump_json(
                            exclude_none=True
                        )
                    }
                )

        print(
            f"[DemoAgent] returning to: {AgentTypeEnum.Supervisor.value}, state: {{'conversationId': {conversation_id}}}"
        )

        print("这是我传递给demo2的context：" + state["context"])
        return Command(
            goto=AgentTypeEnum.Supervisor.value,
            update=await command_update(state),
        )
