import json
import logging
import os
from typing import Type, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.tools import tool

# --- 导入你提供的原始工具函数 ---
from src.tools.ArchiveDecompressionTool import decompress_and_create_replica
from src.tools.FileTreeAnalysisTool import analyze_file_tree
from src.tools.MetaFileReadTool import read_file_metadata


# === 工具 1: 解压与复制工具 ===
class DecompressToolInput(BaseModel):
    source_path: str = Field(description="需要处理的源文件或目录的路径。")
    target_dir: str = Field(description="用于存放处理后文件的工作目录路径。")


@tool("decompress_and_create_replica_tool", args_schema=DecompressToolInput)
def decompress_and_create_replica_tool(source_path: str, target_dir: str) -> str:
    """
    一个用于准备工作目录的工具。它会复制源文件/目录到目标工作区，并自动、递归地解压其中所有支持的压缩文件（如 .zip, .rar, .tar.gz 等）。
    当接收到输入时，应该首先调用此工具以备份文件，避免在原始输入文件上处理。
    返回处理完成后的工作目录路径。
    """
    logging.info(
        f"工具 'decompress_and_create_replica_tool' 被调用，输入: source_path='{source_path}', target_dir='{target_dir}'")
    try:
        processed_path = decompress_and_create_replica(source_path=source_path, target_dir=target_dir)
        return f"成功创建并解压了工作目录，路径是: {processed_path}"
    except Exception as e:
        logging.error(f"解压工具执行失败: {e}")
        return f"错误: 解压和创建副本时失败: {e}"


# === 工具 2: 文件树分析工具 ===
class AnalyzeTreeInput(BaseModel):
    dataset_root_dir: str = Field(description="需要分析的根目录路径。通常是解压工具处理后的工作目录。")
    output_dir: str = Field(description="用于存放中间文件的目录。")
    max_files_to_list: int = Field(default=10, description="在总结同类型文件之前，最多列出的文件数量。")


@tool("analyze_file_tree_tool", args_schema=AnalyzeTreeInput)
def analyze_file_tree_tool(dataset_root_dir: str, output_dir: str, max_files_to_list: int = 10) -> Dict[str, Any]:
    """
    分析指定目录的文件和文件夹结构，并生成一个JSON格式的树状表示。
    这个工具帮助我们宏观地理解数据集的整体结构。
    它会返回一个包含文件树结构的JSON对象。
    """
    logging.info(f"工具 'analyze_file_tree_tool' 被调用，输入: dataset_root_dir='{dataset_root_dir}'")
    try:
        # 注意：原始函数会保存文件，但我们在这里更关心其返回值
        tree_structure = analyze_file_tree(dataset_root_dir=dataset_root_dir, output_dir=output_dir,
                                           max_files_to_list=max_files_to_list)
        if "error" in tree_structure:
            raise RuntimeError(tree_structure["error"])
        return tree_structure
    except Exception as e:
        logging.error(f"文件树分析工具执行失败: {e}")
        return {"error": f"文件树分析失败: {e}"}


# === 工具 3: 文件元数据读取工具 ===
class ReadMetadataInput(BaseModel):
    file_path: str = Field(description="需要读取元数据的单个文件的完整路径。")
    sample_size: int = Field(default=5, description="对于文本或表格类文件，读取并返回前N行作为样本。")


@tool("read_file_metadata_tool", args_schema=ReadMetadataInput)
def read_file_metadata_tool(file_path: str, sample_size: int = 5) -> Dict[str, Any]:
    """
    读取单个文件的元数据。
    对于图片，它会返回尺寸、格式等信息。
    对于CSV、JSON、Parquet等数据文件，它会返回文件的Schema结构和少量数据样本。
    这个工具用于深入了解单个文件的具体内容。
    """
    # 此工具的日志记录在原函数中，这里不再重复
    # logging.info(f"工具 'read_file_metadata_tool' 被调用，输入: file_path='{file_path}'")
    try:
        metadata = read_file_metadata(file_path=file_path, sample_size=sample_size)
        return metadata
    except Exception as e:
        logging.error(f"元数据读取工具执行失败: {e}")
        return {"error": f"读取元数据失败: {e}"}


# === 工具 4: 保存分析结果工具 (新增) ===
class SaveAnalysisInput(BaseModel):
    enhanced_tree_dict: Dict[str, Any] = Field(description="经过AI语义增强后的最终文件树JSON对象。")
    output_dir: str = Field(description="保存最终分析结果的目录路径。")
    filename: str = Field(default="enhanced_analysis.json", description="保存结果的文件名。")


@tool("save_analysis_tool", args_schema=SaveAnalysisInput)
def save_analysis_tool(enhanced_tree_dict: Dict[str, Any], output_dir: str,
                       filename: str = "enhanced_analysis.json") -> str:
    """
    将最终的、经过AI语义增强的JSON分析结果保存到文件中。这是整个流程的最后一步。
    """
    logging.info(f"工具 'save_analysis_tool' 被调用，准备保存文件到 '{output_dir}/{filename}'")
    output_file_path = os.path.join(output_dir, filename)
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_tree_dict, f, ensure_ascii=False, indent=2)
        final_message = f"成功将分析结果保存到: {output_file_path}"
        logging.info(final_message)
        return final_message
    except Exception as e:
        logging.error(f"保存分析工具执行失败: {e}")
        return f"错误: 保存分析结果失败: {e}"
