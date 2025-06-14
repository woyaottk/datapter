"""
JDSecondCoordinator - 一个基于LangGraph的多智能体系统。
该系统实现了一个监督者（Coordinator）管理多个专业代理之间的协作。
"""
import logging
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
from urllib3.filepost import writer

from src.adapter.vo.ai_chat_model import AiChatResultVO
from src.domain.agent.adapter_agent import AdapterAgent
from src.domain.agent.dataset_agent import DatasetAgent
from src.domain.model.model import AdapterState, DatasetAgentState
from src.domain.agent.model_agent import ModelAgent
from src.llm.llm_factory import LLMFactory
from src.llm.model.LLMType import LLMType

SYSTEM_PROMPT = """
# 角色
你是一个模型与数据集适配任务的智能调度助手，负责根据用户意图协调调度三个子代理（DatasetAgent、ModelAgent、AdapterAgent），完成模型代码与数据集的适配任务。

# 工作目标
你的目标是通过理解用户的输入内容和对适配的偏好指令，合理规划和组织以下三个子Agent的协同工作流程：
1. 分析数据集（由 DatasetAgent 执行）
2. 分析模型代码（由 ModelAgent 执行）
3. 生成并执行适配方案（由 AdapterAgent 执行）

# 工作职责
你只负责：
- 理解用户意图和当前任务所处阶段（数据集分析 / 模型分析 / 执行适配 / 结束）
- 调用合适的Agent，并向其传递必要的 user_prompt（用户侧的自然语言说明或偏好）
- 控制任务流程（一般为：先分析 → 后适配），但若用户已有分析结果或跳过分析环节，你应支持跳转执行
- 如用户意图不明确，应主动澄清；如任务完成，路由至 FINISH。

你不负责：
- 分析数据集内容
- 分析模型代码内容
- 执行任何适配推理或代码生成任务

# 子Agent职责
- **DatasetAgent**：分析用户上传的数据集，生成结构、格式、内容的分析报告。
- **ModelAgent**：分析模型代码，识别模型结构、入口函数、数据加载逻辑等。
- **AdapterAgent**：根据 DatasetAgent 与 ModelAgent 的分析结果（JSON报告地址系统已提供），生成模型与数据的适配方案，并执行所需修改。
- **FINISH**：用户显式表示任务结束时路由至此。

# 流程建议
常见标准流程如下：
1. 用户初始上传数据集和代码，未提供分析结果 → 先调用 DatasetAgent 和 ModelAgent，获取分析报告
2. 分析报告就绪 → 向 AdapterAgent 发送分析报告地址，并传递用户可能提出的偏好或约束，生成适配方案
3. 用户验收适配效果，主动声明任务完成 → 路由至 FINISH

但你必须支持用户自定义流程（跳过某步、重新执行某步或追加说明）

# Prompt传递原则
- 你需从用户原始输入中提取关键信息（包括对模型/数据集的描述、对适配方式的偏好、指定模块或目标等），并整理为简洁、具体的 `user_prompt` 传递给子Agent。
- 子Agent所需的技术信息（如上传文件路径、分析结果JSON地址）系统会自动提供，无需你操心。

# 输出格式
1. 你的**思考和意图判断过程**必须使用三引号包裹，如：
\"\"\"当前用户上传了数据集和模型代码，但未提供任何分析结果，因此我将依次调用 DatasetAgent 和 ModelAgent 获取分析信息。\"\"\"
2. 子Agent的调用，必须严格按照以下格式输出：
{format_instructions}
3. 输出应当包括以上两部分，并且中间不要出现其他内容输出。

# 注意事项
- 用户也可能只上传一个文件（只分析数据集或模型），你应按需拆解任务
- 若用户提出“模型中有多个model，目标是适配X模型”或“只想改动数据集不改模型”等偏好，应在 user_prompt 中传达这些要求

"""
# - 如果用户已经明确指出希望“让数据在模型中跑通”、“适配模型”、“生成加载器”等，则说明进入适配阶段，应调度 AdapterAgent


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

    user_prompts: List[str] = Field("传递给每个Agent的用户提示词")


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
            writer = get_stream_writer()
            instruction_string = ""
            reasoning = False
            buffer = ""
            async for chunk in chain.astream(
                    {"format_instructions": suggestion_parser.get_format_instructions()}
            ):
                if chunk.content:
                    buffer += chunk.content
                    split_buf = buffer.split("\"\"\"")
                    reasoning = not reasoning
                    for buf in split_buf:
                        reasoning = not reasoning
                        if reasoning:
                            writer(
                                {
                                    "data": AiChatResultVO(
                                        text=buf
                                    ).model_dump_json(exclude_none=True)
                                }
                            )
                        else:
                            instruction_string += buf
            print("end of agent generate.")
            # response = await chain.ainvoke(
            #     {"format_instructions": suggestion_parser.get_format_instructions()}
            # )
            response = Router.parse_raw(instruction_string)
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
            logging.info(f"Coordinator 选择 Agent To {goto}")
            # 添加判空处理，确保response.answer_type存在并有值
            logging.info(
                f"[Coordinator] isInit=True, goto: {goto}, state: {{'conversationId': {conversation_id}}}"
            )
            if goto == 'DatasetAgent':
                return Command(
                    goto=goto,
                    update={
                        "nextAgents": remaining_agents,
                        "isInit": False,
                        "gotoEnd": False,
                        "dataset_state": DatasetAgentState(input_path=os.getenv("DATASET.INPUT_DIR", "data/input/dataset")),
                    },
                )
            elif goto == 'ModelAgent':
                return Command(
                    goto=goto,
                    update={
                        "nextAgents": remaining_agents,
                        "isInit": False,
                        "model_agent_prompt": response.prompts,
                        "gotoEnd": False,
                        "model_path": os.getenv("CODE.INPUT_DIR", "data/input/code"),
                    },
                )
            else:
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
                logging.info(
                    f"[Coordinator] nextAgents 为空，流转到 END，state: {{'conversationId': {conversation_id}}}"
                )
                return Command(goto=END)

            goto = state["nextAgents"][0]
            remaining_agents = (
                state["nextAgents"][1:] if len(state["nextAgents"]) > 1 else []
            )
            logging.info(
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
        # builder.add_node('ModelAgent', ModelAgent())
        builder.add_node('AdapterAgent', AdapterAgent())

        # 添加边
        builder.add_edge(START, __name__)
        builder.add_edge('DatasetAgent', __name__)
        # builder.add_edge('ModelAgent', __name__)
        builder.add_edge('AdapterAgent', __name__)
        return builder.compile(
            debug=os.getenv("DEBUG", "False").strip().lower() == "true",
        )
