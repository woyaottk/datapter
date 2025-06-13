from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langgraph.config import get_stream_writer
from langgraph.types import Command

from src.adapter.vo.ai_chat_model import AiChatResultVO
from src.domain.constant.constant import AgentTypeEnum
from src.domain.model.model import AdapterState, command_update
from src.llm.llm_factory import LLMFactory
from src.llm.model.LLMType import LLMType

SYSTEM_PROMPT = """
# 角色
你是一个代码适配器智能机器人

# 场景
根据用户的问题进行回答
"""


class Demo2Agent:
    def __init__(self):
        pass

    async def __call__(self, state: AdapterState) -> Command:
        conversation_id = state["conversationId"]
        print(f"[Demo2Agent] called, state: {{'conversationId': {conversation_id}}}")

        print("这是demo传递给我的context：" + state["context"])
        user_question = state["messages"][-1].content

        prompts = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
                ("user", "{user_question}"),
            ]
        )
        llm = await LLMFactory.async_create_llm(LLMType.QWEN)
        chain = prompts | llm

        writer = get_stream_writer()

        async for chunk in chain.astream(
            {
                "user_question": user_question,
            }
        ):
            if chunk.content:
                writer(
                    {
                        "data": AiChatResultVO(text=chunk.content).model_dump_json(
                            exclude_none=True
                        )
                    }
                )

        print(
            f"[Demo2Agent] returning to: {AgentTypeEnum.Supervisor.value}, state: {{'conversationId': {conversation_id}}}"
        )
        return Command(
            goto=AgentTypeEnum.Supervisor.value,
            update=await command_update(state),
        )
