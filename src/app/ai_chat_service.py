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
from src.domain.model.model import DatasetAgentState, AdapterState
from src.utils.SnowFlake import Snowflake

state_cache = {}
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

            init_state = initialize_state(conversation_id, messages)
            # init_state = get_mock_state_4_dataset_agent(messages, conversation_id)
            # init_state = get_mock_state_4_adapter_agent(messages, conversation_id)

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
    # finally:
    #     # 添加完成消息到队列
    #     result = AiChatResultVO(text="")
    #     if result.html is not None:
    #         yield result.model_dump_json(exclude_none=True)


def initialize_state(conversation_id: str, messages: list):
    dataset_state = DatasetAgentState(input_path=os.getenv("DATASET.INPUT_DIR", "data/input/dataset"),
                                      output_path=os.getenv("DATASET.OUTPUT_DIR", "data/output/dataset"),
                                      saved_analysis_filename="",
                                      enhanced_file_tree_json="")
    # with open(os.path.join(os.getenv("CODE.OUTPUT_DIR"), "code_analysis_20250617_065234.json"), "r") as f:
    #     summary = f.read()
    # with open(os.path.join(os.getenv("CODE.OUTPUT_DIR"), "code_analysis_report_20250617_065234.md"), "r") as f:
    #     markdown = f.read()
    # with open(os.path.join(os.getenv("CODE.OUTPUT_DIR"), "code_dataset_info_20250617_065234.json"), "r") as f:
    #     json_out = f.read()
    # model_analyse = {
    #     "markdown": summary,
    #     "json_out": markdown,
    #     "summary": json_out,
    # }
    model_analyse = {
        "markdown": "",
        "json_out": "",
        "summary": "",
    }
    return AdapterState(
        messages=messages,
        isInit=True,
        conversationId=conversation_id,
        messageId=str(sf.generate()),
        nextAgents=[],
        nextPrompts=[],
        model_path=os.getenv("CODE.INPUT_DIR"),
        model_analyse_path=os.getenv("CODE.OUTPUT_DIR"),
        model_analyse=[model_analyse],
        dataset_state=dataset_state,
        dataset_path=os.getenv("DATASET.INPUT_DIR"),
        dataset_analyse="",
        file_operations=[],
    )

def get_mock_state_4_dataset_agent(messages, conversation_id):
    dataset_state = DatasetAgentState(input_path=os.getenv("DATASET.INPUT_DIR", "data/input/dataset"),
                                      output_path=os.getenv("DATASET.OUTPUT_DIR", "data/output/dataset"),
                                      saved_analysis_filename="",
                                      enhanced_file_tree_json="")
    prompt = ""
    return AdapterState(
        messages= messages,
        isInit= False,
        conversationId= conversation_id,
        messageId= str(sf.generate()),
        nextAgents= ['DatasetAgent'],
        nextPrompts= [prompt],
        model_path= os.getenv("CODE.INPUT_DIR"),
        model_analyse_path=os.getenv("CODE.OUTPUT_DIR"),
        model_analyse=None,
        dataset_state= dataset_state,
        context=""
    )

def get_mock_state_4_code_agent(messages, conversation_id):
    prompt = "请分析代码"
    return AdapterState(
        messages= messages,
        isInit= False,
        conversationId= conversation_id,
        messageId= str(sf.generate()),
        nextAgents= ['CodeAgent'],
        nextPrompts= [prompt],
        model_path= os.getenv("CODE.INPUT_DIR"),
        model_analyse_path=os.getenv("CODE.OUTPUT_DIR"),
        model_analyse=None,
        dataset_state= None,
        context=""
    )

def get_mock_state_4_adapter_agent(messages, conversation_id):
    #### mock state
    prompt = "请你根据代码和数据集的分析报告结果，给出将数据集适配至代码的方案。"
    with open(os.path.join(os.getenv("DATASET.OUTPUT_DIR"), "dataset/enhanced_analysis.json"), "r") as f:
        content = f.read()
    dataset_state = DatasetAgentState(enhanced_file_tree_json=content)
    with open(os.path.join(os.getenv("CODE.OUTPUT_DIR"), "code_analysis_20250617_065234.json"), "r") as f:
        summary = f.read()
    with open(os.path.join(os.getenv("CODE.OUTPUT_DIR"), "code_analysis_report_20250617_065234.md"), "r") as f:
        markdown = f.read()
    with open(os.path.join(os.getenv("CODE.OUTPUT_DIR"), "code_dataset_info_20250617_065234.json"), "r") as f:
        json_out = f.read()
    model_analyse = [{
        "markdown": summary,
        "json_out": markdown,
        "summary": json_out,
    }]
    return AdapterState(
        messages= messages,
        isInit= False,
        conversationId= conversation_id,
        messageId= str(sf.generate()),
        nextAgents= ['AdapterAgent'],
        nextPrompts= [prompt],
        model_path= os.getenv("CODE.INPUT_DIR"),
        model_analyse_path=os.getenv("CODE.OUTPUT_DIR"),
        model_analyse=model_analyse,
        dataset_state= dataset_state,
        context=""
    )
