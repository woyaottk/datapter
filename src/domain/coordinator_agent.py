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

from src.domain.dataset_agent import DatasetAgent
from src.domain.model.model import AdapterState
from src.domain.model_agent import ModelAgent
from src.llm.llm_factory import LLMFactory
from src.llm.model.LLMType import LLMType

SYSTEM_PROMPT = """
# 角色
角色  
你是一个适配任务调度助手，专注于识别用户意图并将请求准确路由给对应的分析或适配智能体。

场景  
用户已上传数据集和模型代码，目标是让数据集能够正确运行在模型上。你的任务是：
1. 理解用户的真实意图，明确其当前需求是分析数据集、分析模型，还是完成适配。
2. 调度合适的智能体（Agent）执行任务。
3. 向目标 Agent 传递完整、准确的上下文信息（包括用户的输入、相关历史内容、当前推断出的需求）。
4. 当用户表示任务完成，结束对话。

代理职责  
1. **DatasetAgent**：专注于数据集分析，提供数据集的详细描述。当用户的意图涉及“数据集”时，路由至此代理。
2. **ModelAgent**：专注于模型分析，提供模型的详细描述。当用户的意图涉及“模型”时，路由至此代理。
3. **AdapterAgent**：专注于生成模型与数据集的适配方案，并给出可执行的代码或脚本。当用户希望进行“适配”或提出需要“让数据能跑起来”等类似需求时，路由至此代理。
4. **FINISH**：用户表示任务结束时使用。

工作流程  
1. 识别用户当前的意图和任务阶段（分析 or 适配 or 结束）。  
2. 根据意图，选择合适的代理，并传递上下文信息（包括用户上传的文件、之前的分析结果、用户的问题和说明等）。  
3. 不对分析结果或适配逻辑进行加工或生成，只负责调度。  
4. 在不确定用户意图时，主动提问澄清。

# 输出要求  
1. 必须说明你的意图识别过程和理由。  
2. 必须展示传递给每个代理的上下文信息。  
3. 按照以下格式输出：

# 输出定义:
```
{}
```
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
            'DatasetAgent',
            'ModelAgent',
            'AdapterAgent',
            "FINISH",
        ]
    ] = Field("下一个要使用的Agent序列")

    prompts: List[str] = Field("传递给每个Agent的提示词")

    # description: str = Field("给用户看的描述信息")


class CoordinatorAgent:
    # 定义一个常量, 允许外部访问到该常量
    def __init__(self):
        self.llm = LLMFactory.create_llm(LLMType.QWEN)
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
            goto = response.nextAgents[0]
            # goto如果等于FINISH，则return 使用slotAgent
            if goto == "FINISH":
                goto = __name__
            if goto == 'AdapterAgent':
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
        builder.add_node('DatasetAgent', DatasetAgent())
        builder.add_node('ModelAgent', ModelAgent())
        # builder.add_node('AdapterAgent', )

        # 添加边
        builder.add_edge(START, __name__)
        builder.add_edge('ModelAgent', __name__)
        builder.add_edge('DatasetAgent', __name__)
        # builder.add_edge('AdapterAgent', __name__)
        return builder.compile(
            debug=os.getenv("DEBUG", "False").strip().lower() == "true",
        )
