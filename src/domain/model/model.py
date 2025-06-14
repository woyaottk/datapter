from langgraph.graph import MessagesState
from typing import Optional

class DatasetAgentState(MessagesState): # TODO 新增state，未在其他地方适配
    """数据集Agent返回状态"""
    input_path: Optional[str]
    output_path: Optional[str]
    saved_analysis_filename: Optional[str] # 增强文件树json文件名
    enhanced_file_tree_json: str # 增强后文件树内容


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
    model_agent_prompt = []
    dataset_state: DatasetAgentState = None


async def command_update(state):
    return {
        "nextAgents": state["nextAgents"],
        "messages": state["messages"],
        "context": state["context"],
    }
