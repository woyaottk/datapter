"""
JDSecondCoordinator - 一个基于LangGraph的多智能体系统。
该系统实现了一个监督者（Coordinator）管理多个专业代理之间的协作。
"""

import os

# 标准库导入
from typing import Literal, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langgraph.config import get_stream_writer
from langgraph.constants import END

# 第三方库导入
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from pydantic import Field, BaseModel

from src.domain.demo2_agent import Demo2Agent
from src.domain.demo_agent import DemoAgent
from src.domain.model.model import AdapterState
from src.utils.llm_util import create_llm

SYSTEM_PROMPT = """
# 角色
你是一个代码适配器路由智能机器人

# 场景
负责判断用户输入类型并管理代理调度，根据用户的请求类型和内容，确定处理策略。

# 代理职责
- DemoAgent: 当提示词出现了code时路由到这里
- Demo2Agent: 当提示词出现了model时路由到这里
- FINISH: 结束对话

# 输出定义:
{format_instructions}
"""


class Router(BaseModel):
    """用于确定下一个工作者的路由器。
    Attributes:
        next: 下一个工作者的名称，或者如果任务完成则为FINISH
    """

    # 使用显式类型，避免IDE类型检查器问题
    nextAgents: List[
        Literal[
            DemoAgent.__name__,
            Demo2Agent.__name__,
            "FINISH",
        ]
    ] = Field("下一个要使用的Agent序列")


class CoordinatorAgent:
    # 定义一个常量, 允许外部访问到该常量
    def __init__(self):
        self.llm = create_llm(
            **{
                "model_name": os.getenv("QWEN"),
                "api_key": os.getenv("ALIBABA_API_KEY"),
                "api_base": os.getenv("ALIBABA_BASE_URL"),
                "temperature": 0.2,
            }
        )
        pass

    async def create_agent_func(self, state: AdapterState):
        """监督者节点，决定下一个应该执行的工作者。

        Args:
            state: 当前状态对象

        Returns:
            Command: 包含下一个目标节点和状态更新的命令
        """

        conversation_id = state["conversationId"]

        if state["isInit"]:
            suggestion_parser = PydanticOutputParser(pydantic_object=Router)
            prompts = ChatPromptTemplate.from_messages(
                [SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)]
            )
            # state["messages"] 获取后1条数据，如果不过1条，获取全部数据
            if len(state["messages"]) > 1:
                last_1_messages = state["messages"][-1:]
            else:
                last_1_messages = state["messages"]

            prompts += last_1_messages
            chain = prompts | self.llm | suggestion_parser
            response = await chain.ainvoke(
                {"format_instructions": suggestion_parser.get_format_instructions()}
            )
            print(response)
            goto = response.nextAgents[0]
            # goto如果等于FINISH，则return 使用slotAgent
            if goto == "FINISH":
                goto = __name__

            remaining_agents = (
                response.nextAgents[1:] if len(response.nextAgents) > 1 else []
            )
            # remaining_agents 追加 state["nextAgents"] 确保顺序，并去重
            remaining_agents = list(set(remaining_agents + state["nextAgents"]))
            print(f"Coordinator 选择 Agent To {goto}")
            # 添加判空处理，确保response.answer_type存在并有值
            print(
                f"[Coordinator] isInit=True, goto: {goto}, state: {{'conversationId': {conversation_id}}}"
            )
            return Command(
                goto=goto,
                update={
                    "nextAgents": remaining_agents,
                    "isInit": False,
                    "context": "空",
                    "gotoEnd": False,
                },
            )

        if "nextAgents" in state:
            # 如果state["nextAgents"]=[]，则return END
            if not state["nextAgents"]:
                writer = get_stream_writer()
                writer("")
                print(
                    f"[Coordinator] nextAgents 为空，流转到 END，state: {{'conversationId': {conversation_id}}}"
                )
                return Command(goto=END)

            goto = state["nextAgents"][0]
            remaining_agents = (
                state["nextAgents"][1:] if len(state["nextAgents"]) > 1 else []
            )
            print(
                f"[Coordinator] nextAgents流转，goto: {goto}, remaining_agents: {remaining_agents}, state: {{'conversationId': {conversation_id}}}"
            )
            # 获取当前状态的answer_type，如果不存在则使用默认值
            return Command(
                goto=goto,
                update={
                    "nextAgents": remaining_agents,
                    "gotoEnd": False,
                },
            )
        return None

    def build_coordinator_agent(self):
        # 添加节点
        builder = StateGraph(AdapterState)
        builder.add_node(__name__, self.create_agent_func)
        builder.add_node(DemoAgent.__name__, DemoAgent())
        builder.add_node(Demo2Agent.__name__, Demo2Agent())

        # 添加边
        builder.add_edge(START, __name__)
        builder.add_edge(DemoAgent.__name__,__name__)
        builder.add_edge(Demo2Agent.__name__,__name__)
        return builder.compile(
            debug=os.getenv("DEBUG", "False").strip().lower() == "true",
        )
