import asyncio
import json
import logging
import os
import shutil
from functools import partial
from typing import TypedDict, List, Dict, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- 项目内部模块导入 ---
# 确保这些模块可以被正确找到
from src.adapter.vo.ai_chat_model import AiChatResultVO
from src.domain.constant.constant import AgentTypeEnum
from src.domain.model.model import command_update, AdapterState
from src.llm.llm_factory import LLMFactory
from src.llm.model.LLMType import LLMType

# --- 导入我们新定义的工具 ---
from src.tools.refactored_tools import (
    decompress_and_create_replica_tool,
    analyze_file_tree_tool,
    read_file_metadata_tool,
    save_analysis_tool,
)

# --- 导入原始代码中的 Pydantic 模型和辅助函数 ---
# 假设此文件存在且路径正确
try:
    from dataset_agent import EnhancedFileTree, _build_enhancement_prompt, _extract_summary_statistics
except ImportError:
    try:
        from .dataset_agent import EnhancedFileTree, _build_enhancement_prompt, _extract_summary_statistics
    except (ImportError, ModuleNotFoundError):
        # 提供一个备用方案，以防文件不存在，这有助于隔离问题
        logging.error("无法导入 'dataset_agent'。请确保该文件存在且路径正确。将使用虚拟定义。")
        from pydantic import BaseModel


        class EnhancedFileTree(BaseModel):
            pass


        def _build_enhancement_prompt(a, b, c):
            return "Dummy Prompt"


        def _extract_summary_statistics(a):
            return {}

# 将所有工具放入一个列表中
tools = [
    decompress_and_create_replica_tool,
    analyze_file_tree_tool,
    read_file_metadata_tool,
    save_analysis_tool,
]


# === 1. 定义新的 Agent 状态 (State) ===
# State是图的“内存”，它在所有节点之间传递
class DatasetAgentState(TypedDict):
    input_path: str  # 初始输入路径
    working_dir: str  # 解压和处理后的工作目录
    raw_file_tree: Dict[str, Any]  # 未经处理的文件树结构
    metadata_enriched_tree: Dict[str, Any]  # 添加了元数据的文件树
    final_enhanced_tree: Dict[str, Any]  # 最终由LLM增强的文件树
    messages: List[BaseMessage]  # 对话历史，用于驱动Agent决策
    error_msg: str  # 记录错误信息


# === 2. 定义图的节点 (Nodes) ===

# --- Agent 决策节点 ---
async def agent_node(state: DatasetAgentState, llm: BaseLanguageModel):
    """
    Agent的核心决策节点。根据当前状态和历史消息，决定下一步调用哪个工具或触发哪个阶段。
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
             "你是一个数据集分析专家。你的任务是分析用户提供的数据集路径，并生成一个增强的文件结构报告。"
             "严格按照以下逻辑和指令行动：\n"
             "1. 首先，使用 `decompress_and_create_replica_tool` 和 `analyze_file_tree_tool` 来获取基本的文件结构。工具的输出结果会返回给你。\n"
             "2. 当你收到了 `analyze_file_tree_tool` 的成功结果（一个文件树JSON），你的下一步是富集元数据。要触发这个阶段，请清空你的思考过程，只回复 'ENRICH_METADATA' 这一个词，不要加任何其他内容。\n"
             "3. 当元数据富集完成后，结果会返回给你。你的下一步是进行AI语义增强。要触发这个阶段，请只回复 'ENHANCE_TREE' 这一个词，不要加任何其他内容。\n"
             "4. 当AI语义增强完成后，结果会返回给你。你的下一步是调用 `save_analysis_tool` 来保存最终结果。\n"
             "5. 在确认保存成功后，回复 'FINISH' 来结束任务。"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    llm_with_tools = llm.bind_tools(tools)
    chain = prompt_template | llm_with_tools
    response = await chain.ainvoke({"messages": state["messages"]})
    # The response is already a list containing one message
    return {"messages": state["messages"] + [response]}


# --- 元数据富集节点 ---
async def metadata_enricher_node(state: DatasetAgentState):
    """
    这是一个特殊的“代码”节点，负责遍历文件树并为每个文件调用 'read_file_metadata_tool'。
    """
    logging.info("--- 节点: metadata_enricher_node ---")
    tree_copy = json.loads(json.dumps(state["raw_file_tree"]))
    working_dir = state["working_dir"]
    sample_size = 5

    async def traverse_and_enrich(node: Dict[str, Any], current_path: str):
        node_abs_path = os.path.join(current_path, node["name"])
        if node.get("type") == "file":
            logging.info(f"  正在读取元数据: {node_abs_path}")
            metadata = await asyncio.to_thread(read_file_metadata_tool.func, file_path=node_abs_path,
                                               sample_size=sample_size)
            if metadata:
                node["metadata"] = metadata
        if node.get("type") == "directory" and "children" in node and node["children"]:
            tasks = [traverse_and_enrich(child, node_abs_path) for child in node["children"]]
            await asyncio.gather(*tasks)

    root_node = tree_copy.get("root")
    if root_node:
        await traverse_and_enrich(root_node, os.path.dirname(working_dir))

    # FIX: Use a HumanMessage to report completion. This avoids the ToolMessage error.
    completion_message = HumanMessage(
        content="所有文件的元数据都已成功富集。'metadata_enriched_tree' 字段已填充。请根据指令进行下一步。"
    )
    return {
        "metadata_enriched_tree": tree_copy,
        "messages": state["messages"] + [completion_message],  # Append to history
    }


# --- 最终语义增强节点 ---
async def final_enhancer_node(state: DatasetAgentState, llm: BaseLanguageModel):
    """
    使用LLM进行最终的语义增强。
    """
    logging.info("--- 节点: final_enhancer_node ---")
    metadata_enriched_tree = state["metadata_enriched_tree"]

    parser = PydanticOutputParser(pydantic_object=EnhancedFileTree)
    format_instructions = parser.get_format_instructions()
    summary_stats = _extract_summary_statistics(metadata_enriched_tree)
    prompt = _build_enhancement_prompt(metadata_enriched_tree, summary_stats, format_instructions)

    response_message = await llm.ainvoke(prompt)

    try:
        enhanced_tree_obj = parser.parse(response_message.content)
        final_tree_dict = enhanced_tree_obj.model_dump()
        # FIX: Use a HumanMessage to report completion. This avoids the ToolMessage error.
        completion_message = HumanMessage(
            content="AI语义增强完成。'final_enhanced_tree' 字段已填充。请根据指令进行下一步。"
        )
        return {
            "final_enhanced_tree": final_tree_dict,
            "messages": state["messages"] + [completion_message],  # Append to history
        }
    except Exception as e:
        logging.error(f"解析最终结果失败: {e}", exc_info=True)
        return {"error_msg": f"解析最终增强树时出错: {e}"}


# === 3. 定义图的路由器 (Router) ===
def router(state: DatasetAgentState) -> str:
    """
    根据Agent的最新回复，决定下一个要执行的节点。
    """
    last_message = state["messages"][-1]

    content = getattr(last_message, 'content', '')
    if isinstance(content, str):
        if "ENRICH_METADATA" in content:
            return "enrich_metadata"
        if "ENHANCE_TREE" in content:
            return "enhance_tree"
        if "FINISH" in content:
            return END

    if last_message.tool_calls:
        return "tool_executor"

    return "agent"


# === 4. 主智能体类和图的构建 ===
class DatasetAgent:
    def __init__(self):
        self.llm = LLMFactory.create_llm(LLMType.QWEN)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建LangGraph计算图"""
        graph = StateGraph(DatasetAgentState)

        # FIX: Use functools.partial to pass dependencies to nodes, avoiding asyncio.run()
        agent_with_llm = partial(agent_node, llm=self.llm)
        final_enhancer_with_llm = partial(final_enhancer_node, llm=self.llm)

        graph.add_node("agent", agent_with_llm)
        graph.add_node("tool_executor", ToolNode(tools))
        graph.add_node("enrich_metadata", metadata_enricher_node)
        graph.add_node("enhance_tree", final_enhancer_with_llm)

        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            router,
            {
                "enrich_metadata": "enrich_metadata",
                "enhance_tree": "enhance_tree",
                "tool_executor": "tool_executor",
                "agent": "agent",
                END: END,
            },
        )

        # ToolNode appends the ToolMessage to the state's `messages` list automatically.
        # We need to ensure our agent node does the same.
        graph.add_edge("tool_executor", "agent")
        graph.add_edge("enrich_metadata", "agent")
        graph.add_edge("enhance_tree", "agent")

        return graph.compile()

    async def __call__(self, global_state: AdapterState) -> Dict[str, Any]:
        """LangGraph框架调用的主入口点"""
        dataset_state_dict = global_state.get("dataset_state", {})
        input_path = dataset_state_dict.get("input_path", ".")
        base_name = os.path.basename(input_path.rstrip("/\\")).split(".")[0] if input_path else "default"
        output_dir = os.path.join("output", base_name)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        initial_state = DatasetAgentState(
            input_path=input_path,
            working_dir=output_dir,
            raw_file_tree=None,
            metadata_enriched_tree=None,
            final_enhanced_tree=None,
            messages=[HumanMessage(
                content=f"请分析位于 '{input_path}' 的数据集。请将所有输出和中间文件都放入 '{output_dir}' 目录。")],
            error_msg=None
        )

        final_graph_state = None
        try:
            async for event in self.graph.astream_events(initial_state, version="v1"):
                kind = event["event"]
                if kind == "on_chain_end":
                    if "output" in event["data"]:
                        final_graph_state = event["data"]["output"]
                logging.info(f"--- Agent Event: {kind} ---\n{event['data']}\n---")

            if not final_graph_state:
                raise RuntimeError("图执行流未能正常结束并返回最终状态。")

            final_tree = final_graph_state.get("final_enhanced_tree")
            error_message = final_graph_state.get("error_msg")
            if error_message:
                raise RuntimeError(f"图执行过程中出现错误: {error_message}")
            if not final_tree:
                raise RuntimeError("图执行完毕，但未能生成最终的增强文件树。")

            json_string = json.dumps(final_tree, ensure_ascii=False, indent=2)
            dataset_state_dict['output_path'] = output_dir
            dataset_state_dict['saved_analysis_filename'] = "enhanced_analysis.json"
            dataset_state_dict['enhanced_file_tree_json'] = json_string
            dataset_state_dict.pop('error_msg', None)

        except Exception as e:
            error_message = f"执行过程中发生严重错误: {e}"
            logging.error(error_message, exc_info=True)
            dataset_state_dict['error_msg'] = error_message
        finally:
            global_state['dataset_state'] = dataset_state_dict
            return global_state


# --- 为了能独立运行和测试，添加一个简单的main函数 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

    mock_global_state = {
        "dataset_state": {
            "input_path": r"D:\Code\datapter\data\input\input_dataset.zip"
        }
    }

    agent = DatasetAgent()
    final_state = asyncio.run(agent(mock_global_state))

    print("\n\n===== FINAL RESULT =====")
    print(json.dumps(final_state, indent=2, ensure_ascii=False))
