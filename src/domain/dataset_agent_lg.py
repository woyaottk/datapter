import asyncio
import json
import os
import shutil
from typing import List, Optional, Dict, Any

# --- LangChain/LangGraph 核心库导入 ---
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseLanguageModel
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.domain.constant.constant import AgentTypeEnum
# --- 项目内部模块导入 ---
from src.domain.model.model import DatasetAgentState
from src.domain.model.model import command_update, AdapterState
from src.llm.llm_factory import LLMFactory
from src.llm.model.LLMType import LLMType
from src.tools.ArchiveDecompressionTool import decompress_and_create_replica
from src.tools.FileTreeAnalysisTool import analyze_file_tree
from src.tools.MetaFileReadTool import read_file_metadata


# === 1. Pydantic 数据结构定义 (Agent内部使用) ===

class EnhancedFileNode(BaseModel):
    name: str = Field(description="文件或目录的名称。")
    type: str = Field(description="节点类型：'file', 'directory', 或 'summary'。")
    size: Optional[int] = Field(default=None, description="文件大小（字节）。")
    children: Optional[List["EnhancedFileNode"]] = Field(default=None, description="子节点列表。")
    # new feature: add meta data info
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="从文件中提取的元数据，如图像尺寸或CSV列名。")

    description: Optional[str] = Field(default=None, description="节点描述。")
    count: Optional[int] = Field(default=None, description="省略表示的文件总数。")
    extension: Optional[str] = Field(default=None, description="省略表示的文件扩展名。")
    format: Optional[str] = Field(default=None, description="文件格式/扩展名。")
    is_data_container: Optional[bool] = Field(default=None, description="是否为核心数据容器。")
    purpose_tag: Optional[str] = Field(default=None, description="主要用途标签。")
    data_type_suggestion: Optional[str] = Field(default=None, description="建议的数据类型。")
    relevance_group_id: Optional[str] = Field(default=None, description="功能相关节点组的标识符。")


class EnhancedFileTree(BaseModel):
    root: EnhancedFileNode = Field(description="增强文件树的根节点。")


# === 2. 辅助函数与配置 ===

def load_config():
    """从环境变量加载配置。"""
    from dotenv import load_dotenv
    try:
        # 确保.env文件路径相对于您运行脚本的位置是正确的
        load_dotenv(r'..\..\.env')
    except Exception:
        print("Warning: .env file not found. Using default environment variables.")

    return {
        "max_files_to_list": int(os.getenv("DATASET.MAX_FILES_PER_TYPE_TO_LIST", 10)),
        "output_dir": os.getenv("DATASET.OUTPUT_DIR", "output"),
        "sample_size": int(os.getenv("DATASET.SAMPLE_SIZE", 5)),  # 用于元数据采样
    }


def _extract_summary_statistics(file_tree: Dict[str, Any]) -> Dict[str, Any]:
    """遍历文件树，提取关于省略节点的统计信息。"""
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


def _build_enhancement_prompt(file_tree: Dict[str, Any], summary_stats: Dict[str, Any],
                              format_instructions: str) -> str:
    """构建用于AI增强的提示，包含文件元数据上下文和更严格的指令。"""
    # --- 修正了对 summary 节点的解释 ---
    summary_line = (
        "**极其重要的指令：关于'省略表示'的正确理解**\n"
        "- 文件树中的`type: \"summary\"`节点用于表示一个文件夹中大量的同类型文件。\n"
        "- 一个`summary`节点中的`count`字段，例如`\"count\": 166`，代表该目录下此类文件的 **绝对总数**。\n"
        "- **请绝对不要将`summary`节点旁边列出的几个文件示例的数量与`count`字段相加**。如果看到5个`.png`文件示例和一个`count: 166`的`.png`摘要，这意味着总共有166个`.png`文件，而不是171个。错误地相加会严重影响分析结果。"
    )

    # --- 更新了对新元数据格式的解释 ---
    metadata_instructions = (
        "**元数据可用性与格式说明**:\n"
        "- 文件节点中的`metadata`字段包含两种格式：\n"
        "  1. **`{\"type\": \"structured\", \"data\": {...}}`**: 用于非文本文件（如图片）。`data`字段包含结构化的键值对（如`width`, `height`）。请直接利用这些信息。\n"
        "  2. **`{\"type\": \"raw_sample\", \"content\": \"...\"}`**: 用于文本文件（如CSV, Parquet, JSON）。`content`字段包含文件开头的原始文本样本。**你需要自行分析这段文本**来推断其结构、列名、数据类型等信息。\n"
        "- 如果`metadata`包含`error`字段，说明文件读取失败，你可以基于此进行标注。"
    )

    return f"""你是一个顶尖的数据集文件结构分析专家。你的任务是分析输入的JSON格式文件树，并返回一个结构完全相同但语义增强的文件树。
{summary_line}
{metadata_instructions}
**核心指令:**
1. **精确理解元数据**: 严格按照上述两种格式解读`metadata`字段，对`raw_sample`内容进行深入分析。
2. **保持原始结构**: 绝对不能修改 `name`, `type`, `size`, `children`, `count`, `extension`, `metadata` 等原始字段。
3. **全面语义标注**: 为每个节点添加 `description`, `format`, `is_data_container`, `purpose_tag`, `data_type_suggestion`, `relevance_group_id` 字段。
**输出格式:**
{format_instructions}
**输入文件树:**
{json.dumps(file_tree, ensure_ascii=False, indent=2)}
请基于以上信息，返回增强后的文件树JSON。"""


# === 3. 主要智能体类 ===

class DatasetAgent:
    def __init__(self):
        """初始化智能体。"""
        pass

    async def __call__(self, global_state: AdapterState) -> Command:
        """
        LangGraph框架调用的主入口点。
        以批处理模式执行完整的分析流程，并返回一个包含最终输出字段的字典。
        """
        state = None
        try:
            # --- 初始化 ---
            config = load_config()
            llm = LLMFactory.create_llm(LLMType.QWEN)
            state = global_state.get("dataset_state")
            input_path = state.get("input_path")
            if not input_path or not await asyncio.to_thread(os.path.exists, input_path):
                raise ValueError(f"输入路径无效或不存在: {input_path}")

            # --- 按新的 5 个阶段顺序执行 ---
            output_dir, processed_dir = await self._run_stage_1_setup(input_path, config, decompress_and_create_replica)
            raw_file_tree = await self._run_stage_2_analyze_tree(processed_dir, output_dir, config, analyze_file_tree)
            metadata_enriched_tree = await self._run_stage_3_read_metadata(raw_file_tree, processed_dir, config,
                                                                           read_file_metadata)
            enhanced_tree_dict = await self._run_stage_4_enhance_tree(metadata_enriched_tree, llm)
            filename, json_string = await self._run_stage_5_save_results(enhanced_tree_dict, output_dir)

            # --- 成功返回，填充状态字典 ---
            print("[DatasetAgent] 运行成功, 返回最终状态。")
            state['output_path'] = output_dir
            state['saved_analysis_filename'] = filename
            state['enhanced_file_tree_json'] = json_string
            global_state['dataset_state'] = state
            return Command(
                goto=AgentTypeEnum.Supervisor.value,
                update=await command_update(global_state),
            )
        except Exception as e:
            error_message = f"执行过程中发生严重错误: {e}"
            print(f"[DatasetAgent] {error_message}")
            if not state:
                state = DatasetAgentState()
            state['error_msg'] = error_message
            global_state['dataset_state'] = state
            return Command(
                goto=AgentTypeEnum.Supervisor.value,
                update=await command_update(global_state),
            )

    async def _run_stage_1_setup(self, input_path: str, config: dict, decompress_func) -> (str, str):
        """阶段 1: 设置工作目录并解压源文件。"""
        print("--- 阶段 1: 设置与解压 ---")
        base_name = os.path.basename(input_path.rstrip("/\\")).split(".")[0]
        run_output_dir = os.path.join(config["output_dir"], base_name)

        if await asyncio.to_thread(os.path.exists, run_output_dir):
            await asyncio.to_thread(shutil.rmtree, run_output_dir)
        await asyncio.to_thread(os.makedirs, run_output_dir, exist_ok=True)

        processed_path = await asyncio.to_thread(
            decompress_func,
            source_path=input_path, target_dir=run_output_dir
        )
        return run_output_dir, processed_path

    async def _run_stage_2_analyze_tree(self, processed_dir: str, output_dir: str, config: dict, analyze_func) -> Dict[
        str, Any]:
        """阶段 2: 分析文件树结构。"""
        print("--- 阶段 2: 分析文件树 ---")
        result = await asyncio.to_thread(
            analyze_func,
            dataset_root_dir=processed_dir,
            output_dir=output_dir,
            max_files_to_list=config["max_files_to_list"],
        )
        if "error" in result:
            raise RuntimeError(f"文件树分析失败: {result.get('error')}")
        return result

    async def _run_stage_3_read_metadata(self, raw_file_tree: dict, processed_dir: str, config: dict, read_meta_func) -> \
    Dict[str, Any]:
        """阶段 3: 读取文件元数据并丰富文件树。"""
        print("--- 阶段 3: 读取文件元数据 ---")
        sample_size = config["sample_size"]

        async def traverse_and_enrich(node: Dict[str, Any], current_path: str):
            """递归函数，用于遍历树并调用元数据读取工具。"""
            node_abs_path = os.path.join(current_path, node["name"])

            if node.get("type") == "file":
                metadata = await asyncio.to_thread(read_meta_func, node_abs_path, sample_size=sample_size)
                if metadata:
                    node["metadata"] = metadata

            if node.get("type") == "directory" and "children" in node and node["children"]:
                for child in node["children"]:
                    await traverse_and_enrich(child, node_abs_path)

        tree_copy = json.loads(json.dumps(raw_file_tree))

        root_node = tree_copy.get("root")
        if root_node:
            await traverse_and_enrich(root_node, os.path.dirname(processed_dir))

        meta_tree_path = os.path.join(processed_dir, "metadata_enriched_tree.json")
        with open(meta_tree_path, 'w', encoding='utf-8') as f:
            json.dump(tree_copy, f, ensure_ascii=False, indent=2)
        print(f"带有元数据的文件树已保存至: {meta_tree_path}")
        return tree_copy

    async def _run_stage_4_enhance_tree(self, metadata_enriched_tree: dict, llm: BaseLanguageModel) -> Dict[str, Any]:
        """阶段 4: 使用AI进行语义增强。"""
        print("--- 阶段 4: AI语义增强 ---")
        summary_stats = _extract_summary_statistics(metadata_enriched_tree)

        parser = PydanticOutputParser(pydantic_object=EnhancedFileTree)
        format_instructions = parser.get_format_instructions()
        prompt = _build_enhancement_prompt(metadata_enriched_tree, summary_stats, format_instructions)

        response_message = await llm.ainvoke(prompt)
        enhanced_tree_obj = parser.parse(response_message.content)

        return enhanced_tree_obj.model_dump()

    async def _run_stage_5_save_results(self, enhanced_tree_dict: dict, output_dir: str) -> (str, str):
        """阶段 5: 保存最终结果并完成。"""
        print("--- 阶段 5: 保存并完成 ---")
        filename = "enhanced_analysis.json"
        output_file_path = os.path.join(output_dir, filename)

        def _save_json():
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_tree_dict, f, ensure_ascii=False, indent=2)

        await asyncio.to_thread(_save_json)
        json_string = json.dumps(enhanced_tree_dict, ensure_ascii=False, indent=2)
        print(f"最终结果已保存至: {output_file_path}")
        return filename, json_string
