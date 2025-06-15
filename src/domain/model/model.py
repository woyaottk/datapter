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
    # global（整个会话共享）
    conversationId: str
    conversation_id: str
    message_id: str
    blockId: str
    messageId: str
    isInit: bool = True
    nextAgents: list[str] = []  # 剩余 agent
    nextPrompts: list[str] = [] # 剩余 prompt
    messages: list[dict] = []   # 历史消息
    # agent（单次agent调用）
    prompt: str                 # 传递给当前agent的prompt
    context: str                # 其他必要的上下文信息
    # model 相关
    model_path:str              # 模型代码输入路径
    model_analyse:list[dict] = []   #  模型分析结果
    model_agent_prompt:list = []    # todo: use prompt instead of model_agent_prompt
    # dataset 相关
    dataset_path:str            # 数据集输入路径
    dataset_analyse:str         # json 格式
    dataset_state: DatasetAgentState = None
    # adapter 相关
    file_operations: List[str]  # 文件操作结果


async def command_update(state):
    return {
        "nextAgents": state["nextAgents"],
        "messages": state["messages"],
        "context": state["context"],
    }
