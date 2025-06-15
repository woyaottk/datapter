import asyncio
import logging
import os
import sys
from pathlib import Path
import json

from langchain_core.messages import HumanMessage
from sse_starlette.sse import AsyncContentStream, logger

from src.adapter.vo.ai_chat_model import ChatInputVO, AiChatResultVO
from src.domain.agent.coordinator_agent import CoordinatorAgent
from src.utils.SnowFlake import Snowflake


sf = Snowflake(worker_id=0, datacenter_id=0)
# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


async def chat_handler(request: ChatInputVO) -> AsyncContentStream:
    conversation_id = request.conversation_id
    try:
        try:
            graph = CoordinatorAgent().build_coordinator_agent()
            messages = []
            msg = request.messages[-1]
            content = (
                msg.content.get("text", "")
                if isinstance(msg.content, dict)
                else msg.content
            )
            if content:  # 确保content不为空
                messages.append(HumanMessage(content=str(content)))

            init_state = {
                "messages": messages,
                "isInit": True,
                "gotoEnd": False,
                "conversationId": conversation_id,
                "messageId": str(sf.generate()),
                "nextAgents": ['ModelAgent'],# change next
                "nextPrompts": ['请你分析这个模型的数据加载方法和dataloader结构'],
                "model_path": os.getenv("CODE.INPUT_DIR"),
                "model_analyse_path":os.getenv("CODE.OUTPUT_DIR"),
                "model_analyse":[]
            }
            try:
                ai_response_text = ""
                async for chunk in graph.astream(
                    init_state,
                    config={
                        "configurable": {"thread_id": conversation_id},
                    },
                    stream_mode=["custom"],
                    subgraphs=True,
                ):
                    # 优化：添加类型检查和更好的错误处理
                    if not chunk:
                        continue

                    if isinstance(chunk, tuple):
                        start, mode, data = chunk
                        chunk = data

                    # 收集AI响应文本
                    if isinstance(chunk, dict) and "data" in chunk:
                        try:
                            data_dict = json.loads(chunk["data"])
                            if isinstance(data_dict, dict) and "text" in data_dict:
                                ai_response_text += data_dict["text"]
                        except json.JSONDecodeError:
                            logging.error("Error decoding JSON data:", chunk["data"])

                    # 优化：根据不同的stream_mode处理不同类型的数据
                    yield chunk
                logging.info(ai_response_text)
            except asyncio.TimeoutError:
                logging.error("timeout")
        except Exception as e:
            logging.error(e)
            if os.getenv("DEBUG", "false") == "true":
                import traceback
                traceback.print_stack()
                exit(0)

    except Exception as e:
        logging.error(e)
    finally:
        # 添加完成消息到队列
        result = AiChatResultVO(text="")
        if result.html is not None:
            yield result.model_dump_json(exclude_none=True)
