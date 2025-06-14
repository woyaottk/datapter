from langgraph.graph import MessagesState
from typing import Optional, List, Dict, Any


class DatasetAgentState(MessagesState): # TODO 新增state，未在其他地方适配
    """数据集Agent返回状态"""
    input_path: Optional[str]
    output_path: Optional[str]
    saved_analysis_filename: Optional[str] # 增强文件树json文件名
    enhanced_file_tree_json: Optional[str] # 增强后文件树内容
    error_msg: Optional[str]


class CollaborativeAgentState(MessagesState):
    """协作代理系统的状态类"""
    information_summary: str = ""
    need_more_info: bool = True
    current_working_path: str = "./"  # 改为通用的默认路径
    discovered_paths: List[str] = []  # 添加已发现路径列表
    path_context: Dict[str, Any] = {}  # 添加路径上下文信息
    target_base_path: str = "./"  # 添加目标基础路径，用于存储用户指定的根路径


class AdapterState(MessagesState):
    """状态类，继承自MessagesState并添加next属性。

    Attributes:
        next: 跟踪下一个应该执行的节点
    """

    conversationId: str
    conversation_id: str
    message_id: str
    blockId: str
    messageId: str
    isInit: bool = True
    nextAgents: list[str] = []
    messages: list[dict] = []
    context: str
    model_path:str
    model_analyse:list[dict] = []
    model_agent_prompt:list = []
    dataset_state: DatasetAgentState = None
    file_operations: List[str]


async def command_update(state):
    return {
        "nextAgents": state["nextAgents"],
        "messages": state["messages"],
        "context": state["context"],
    }
