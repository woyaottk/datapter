import os
import json
import shutil
from typing import List, Optional, Dict, Any, TypedDict, Annotated
import operator

# --- LangChain/LangGraph 库导入 ---
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseLanguageModel

# --- 自定义工具导入 ---
from src.tools.FileTreeAnalysisTool import analyze_file_tree
from src.tools.ArchiveDecompressionTool import decompress_and_create_replica
from src.llm.llm_factory import LLMFactory
from src.llm.model.LLMType import LLMType
from model.model import DatasetAgentState

# === 配置管理 ===
def load_config():
    """从环境变量加载配置。"""
    from dotenv import load_dotenv
    load_dotenv(r'..\..\.env')
    # 默认值
    config_data = {
        "max_files_to_list": int(os.getenv("DATASET.MAX_FILES_PER_TYPE_TO_LIST", 10)),
        "output_dir": os.getenv("DATASET.OUTPUT_DIR"),
    }
    return config_data


# === Pydantic 模型  ===
class EnhancedFileNode(BaseModel):
    name: str = Field(description="文件或目录的名称。必须保留输入中的原始值。")
    type: str = Field(
        description="节点类型：'file' (文件)、'directory' (目录) 或 'summary' (省略摘要)。必须保留输入中的原始值。")
    size: Optional[int] = Field(default=None,
                                description="文件大小（字节，仅适用于文件）。如果输入中存在，则必须保留原始值。")
    children: Optional[List["EnhancedFileNode"]] = Field(default=None,
                                                         description="子节点列表（仅适用于目录）。必须保留输入中的原始结构和内容。")
    description: Optional[str] = Field(default=None, description="节点描述或省略摘要描述。")
    count: Optional[int] = Field(default=None, description="省略表示的文件总数（仅适用于summary类型）。")
    extension: Optional[str] = Field(default=None, description="省略表示的文件扩展名（仅适用于summary类型）。")
    format: Optional[str] = Field(default=None,
                                  description="文件格式/扩展名 (例如, 'jpg', 'parquet', 'txt')。对于目录和summary，此值为 null。")
    is_data_container: Optional[bool] = Field(default=None,
                                              description="布尔值，表示此节点是否主要存储数据集的核心数据。如果未知，则使用 null。")
    purpose_tag: Optional[str] = Field(default=None,
                                       description="主要用途标签：'training_data', 'validation_data', 'test_data', 'annotations', 'documentation', 'scripts', 'auxiliary', 'unknown'。如不适用，则使用 null。")
    data_type_suggestion: Optional[str] = Field(default=None,
                                                description="建议的数据类型：'images', 'text_corpus', 'tabular_data', 'audio', 'video', 'mixed', 'unknown'。如不适用，则使用 null。")
    relevance_group_id: Optional[str] = Field(default=None,
                                              description="功能相关节点组的标识符。如果节点不属于任何组或未知，则使用 null。")


class EnhancedFileTree(BaseModel):
    root: EnhancedFileNode = Field(
        description="增强文件树的根节点。此根节点及其子节点的结构必须与输入文件树结构完全匹配。")


# === 工作流状态定义  ===
class DatasetAnalysisState(TypedDict):
    input_path: str
    run_output_dir: str
    processed_replica_dir: str
    raw_file_tree: Optional[Dict[str, Any]]
    enhanced_file_tree: Optional[Dict[str, Any]]
    error: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    analysis_metadata: Optional[Dict[str, Any]]


# === 辅助函数 ===
def _extract_summary_statistics(file_tree: Dict[str, Any]) -> Dict[str, Any]:
    stats = {"total_summary_nodes": 0, "total_omitted_files": 0, "summary_by_extension": {}}

    def _traverse(node):
        if isinstance(node, dict):
            if node.get("type") == "summary":
                stats["total_summary_nodes"] += 1
                count = node.get("count", 0)
                ext = node.get("extension", "unknown")
                stats["total_omitted_files"] += count
                stats["summary_by_extension"].setdefault(ext, {"count": 0, "total_files": 0})
                stats["summary_by_extension"][ext]["count"] += 1
                stats["summary_by_extension"][ext]["total_files"] += count
            for child in node.get("children", []):
                _traverse(child)

    _traverse(file_tree.get("root", {}))
    return stats


def _build_enhancement_prompt(file_tree: Dict[str, Any], metadata: Dict[str, Any], format_instructions: str) -> str:
    summary_stats = metadata.get("summary_statistics", {})
    return f"""你是一个数据集文件结构分析智能体。你的任务是分析输入的JSON格式文件树，并返回一个结构完全相同但语义增强的文件树。
**重要提示：此文件树使用了"省略表示"功能**
- 当文件夹中有大量同类型文件时，只显示前几个文件，其余用 `type: "summary"` 的节点表示。
- 本次分析发现了 {summary_stats.get('total_summary_nodes', 0)} 个省略节点，共省略了 {summary_stats.get('total_omitted_files', 0)} 个文件。
**关键指令:**
1. **保持原始结构**: 绝对不能修改 `name`, `type`, `size`, `children`, `count`, `extension` 等原始字段。
2. **理解省略表示**: 基于 `count` 和 `extension` 推断这些文件的用途和重要性。
3. **全面语义标注**: 为每个节点（包括file, directory, summary）添加 `description`, `format`, `is_data_container`, `purpose_tag`, `data_type_suggestion`, `relevance_group_id` 字段。
**输出格式:**
{format_instructions}
**输入文件树:**
{json.dumps(file_tree, ensure_ascii=False, indent=2)}
请基于以上信息，返回增强后的文件树JSON。"""


# === 主要智能体类 ===
class DatapterAgent:
    def __init__(self, llm: BaseLanguageModel, config: Dict[str, Any]):
        """
        初始化智能体，接收所需的LLM和配置。
        """
        self.llm = llm
        self.config = config
        self.workflow = self._create_workflow()
        print("Datapter Agent 初始化完成")

    # --- 节点函数现在是类的内部方法 ---
    def _setup_and_decompress_node(self, state: DatasetAnalysisState) -> DatasetAnalysisState:
        """阶段 0: 创建统一的输出目录并解压输入文件。"""
        print("\n=== 阶段零：创建输出目录并解压缩 开始 ===")
        input_path = state["input_path"]
        try:
            base_name = os.path.basename(input_path.rstrip("/\\")).split(".")[0]
            base_output_dir = self.config["output_dir"]
            run_output_dir = os.path.join(base_output_dir, base_name)

            if os.path.exists(run_output_dir):
                shutil.rmtree(run_output_dir)
            os.makedirs(run_output_dir, exist_ok=True)
            state["run_output_dir"] = run_output_dir
            print(f"所有输出将保存在统一目录: {run_output_dir}")

            processed_path = decompress_and_create_replica(
                source_path=input_path, target_dir=run_output_dir
            )
            state["processed_replica_dir"] = processed_path
            print(f"阶段零：输入路径处理完成。处理后的副本路径: {processed_path}")
        except Exception as e:
            error_msg = f"在设置或解压阶段发生错误: {e}"
            state["error"] = error_msg
            print(f"阶段零：失败 - {error_msg}")
        print("=== 阶段零：创建输出目录并解压缩 结束 ===")
        return state

    def _analyze_file_tree_node(self, state: DatasetAnalysisState) -> DatasetAnalysisState:
        """阶段 1: 分析文件树结构。"""
        print("\n=== 阶段一：文件树结构分析 开始 ===")
        if state.get("error"):
            print(f"由于先前错误，跳过文件树分析: {state['error']}")
            return state
        try:
            result = analyze_file_tree(
                dataset_root_dir=state["processed_replica_dir"],
                output_dir=state["run_output_dir"],
                max_files_to_list=self.config["max_files_to_list"],
            )
            if "error" not in result:
                state["raw_file_tree"] = result
                print("阶段一：文件树结构分析成功。")
            else:
                state["error"] = f"文件树分析失败: {result['error']}"
                print(f"阶段一：分析失败 - {state['error']}")
        except Exception as e:
            state["error"] = f"文件树分析时发生意外错误: {e}"
            print(f"阶段一：分析失败 - {state['error']}")
        print("=== 阶段一：文件树结构分析 结束 ===")
        return state

    def _enhance_file_tree_node(self, state: DatasetAnalysisState) -> DatasetAnalysisState:
        """阶段 2: 使用LLM对文件树进行语义增强。"""
        print("\n=== 阶段二：文件树语义增强 开始 ===")
        if state.get("error") or not state.get("raw_file_tree"):
            print("由于错误或缺少原始文件树，跳过增强阶段。")
            return state
        try:
            parser = PydanticOutputParser(pydantic_object=EnhancedFileTree)
            format_instructions = parser.get_format_instructions()
            summary_stats = _extract_summary_statistics(state["raw_file_tree"])
            state["analysis_metadata"] = {"summary_statistics": summary_stats}
            prompt = _build_enhancement_prompt(state["raw_file_tree"], state["analysis_metadata"], format_instructions)

            # 使用类实例中的LLM
            response_message = self.llm.invoke(prompt)

            enhanced_tree_obj = parser.parse(response_message.content)
            state["enhanced_file_tree"] = enhanced_tree_obj.model_dump()
            print("阶段二：文件树语义增强成功。")
        except Exception as e:
            state["error"] = f"文件树增强过程失败: {e}"
            print(f"阶段二：失败 - {state['error']}")
        print("=== 阶段二：文件树语义增强 结束 ===")
        return state

    def _save_results_node(self, state: DatasetAnalysisState) -> DatasetAnalysisState:
        """阶段 3: 保存最终的增强文件树。"""
        print("\n=== 阶段三：保存最终分析结果 开始 ===")
        if state.get("enhanced_file_tree") and not state.get("error"):
            try:
                output_file_path = os.path.join(state["run_output_dir"], "enhanced_analysis_results.json")
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(state["enhanced_file_tree"], f, ensure_ascii=False, indent=2)
                print(f"阶段三：最终增强结果已保存至: {output_file_path}")
            except Exception as e:
                state["error"] = f"保存最终结果失败: {e}"
                print(f"阶段三：失败 - {state['error']}")
        else:
            print("由于错误或无有效数据，跳过保存最终结果。")
        print("=== 阶段三：保存最终分析结果 结束 ===")
        return state

    def _create_workflow(self):
        """构建LangGraph工作流，将节点方法连接起来。"""
        workflow = StateGraph(DatasetAnalysisState)
        workflow.add_node("setup_and_decompress", self._setup_and_decompress_node)
        workflow.add_node("analyze_file_tree", self._analyze_file_tree_node)
        workflow.add_node("enhance_file_tree", self._enhance_file_tree_node)
        workflow.add_node("save_results", self._save_results_node)

        workflow.set_entry_point("setup_and_decompress")
        workflow.add_edge("setup_and_decompress", "analyze_file_tree")
        workflow.add_edge("analyze_file_tree", "enhance_file_tree")
        workflow.add_edge("enhance_file_tree", "save_results")
        workflow.add_edge("save_results", END)

        return workflow.compile()

    def analyze_dataset(self, input_path: str) -> Dict[str, Any]:
        """用于运行数据集分析工作流的公开方法。"""
        print(f"\n开始分析数据集: {input_path}")
        if not os.path.exists(input_path):
            return {"error": f"输入路径不存在: {input_path}"}

        initial_state = {"input_path": input_path, "messages": []}
        final_state = self.workflow.invoke(initial_state)

        result = {
            "success": not bool(final_state.get("error")),
            "run_output_dir": final_state.get("run_output_dir"),
            "enhanced_file_tree": final_state.get("enhanced_file_tree"),
            "error": final_state.get("error"),
        }

        if result["success"]:
            print(f"\n✅ 数据集分析成功完成！所有结果保存在: {result['run_output_dir']}")
        else:
            print(f"\n❌ 数据集分析失败: {result['error']}")
        return result


# === 主执行入口 ===
def main():
    # --- 加载配置并初始化LLM ---
    config = load_config()
    llm = LLMFactory.create_llm(LLMType.QWEN)
    print("=" * 60)
    print("Datapter (DA) 智能体 - 数据集结构分析与描述生成")
    print("=" * 60)

    # --- 实例化智能体并传入依赖，然后运行 ---
    agent = DatapterAgent(llm=llm, config=config)
    default_input = os.getenv("DATASET.INPUT_DIR")

    if not default_input:
        print("错误: 请在 .env 文件中设置 'DATASET.INPUT_DIR' 环境变量。")
        return

    # 执行分析
    result = agent.analyze_dataset(default_input)


if __name__ == "__main__":
    main()