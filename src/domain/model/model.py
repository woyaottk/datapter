from langgraph.graph import MessagesState


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


async def command_update(state):
    return {
        "nextAgents": state["nextAgents"],
        "messages": state["messages"],
    }
