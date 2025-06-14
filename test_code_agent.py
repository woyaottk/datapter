import os
from src.domain.model_agent import ModelAgent

from dotenv import load_dotenv
load_dotenv('.env')

async def test_model_agent():
    """
    测试 CodeAgent 的核心逻辑是否能正常运行。
    模拟 chat_handler 中的行为，构建并执行一个 Agent。
    """
    # 1. 准备输入参数（state）
    from src.domain.model.model import AdapterState

    input_path = os.getenv("CODE.INPUT_DIR")

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
        model_path=input_path,
        model_analyse=[],
        model_agent_prompt=[],
    )
    print(global_state)

    # 2. 初始化 Agent
    model_agent = ModelAgent()

    # 3. 调用 Agent 并获取结果
    result = await model_agent(global_state)

    # 4. 打印结果以确认执行成功
    print("Agent 返回结果:", result)


if __name__ == '__main__':
    import asyncio

    # 运行测试方法
    asyncio.run(test_model_agent())