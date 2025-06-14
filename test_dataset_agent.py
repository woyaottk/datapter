import os
from src.domain.dataset_agent_lg import DatasetAgent

from dotenv import load_dotenv
load_dotenv('.env')

async def test_dataset_agent():
    """
    测试 DatasetAgent 的核心逻辑是否能正常运行。
    模拟 chat_handler 中的行为，构建并执行一个 Agent。
    """
    # 1. 准备输入参数（state）
    from src.domain.model.model import DatasetAgentState, AdapterState

    input_path = os.getenv("DATASET.INPUT_DIR")
    dataset_state = DatasetAgentState(
        input_path=input_path
    )

    global_state = AdapterState(
        conversationId='',
        conversation_id='',
        message_id='',
        blockId='',
        messageId='',
        isInit=True,
        nextAgents=[],
        messages=[],
        context='',
        model_path='',
        model_analyse=[],
        model_agent_prompt=[],
        dataset_state=dataset_state
    )
    print(global_state)

    # 2. 初始化 Agent
    datapter_agent = DatasetAgent()

    # 3. 调用 Agent 并获取结果
    result = await datapter_agent(global_state)

    # 4. 打印结果以确认执行成功
    print("Agent 返回结果:", result)


if __name__ == '__main__':
    import asyncio

    # 运行测试方法
    asyncio.run(test_dataset_agent())