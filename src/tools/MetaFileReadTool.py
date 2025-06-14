import logging
import os
import json
import pandas as pd
from PIL import Image
from typing import Dict, Any, List


# --- Metadata Reader for Specific File Types ---

def _read_image_metadata(file_path: str, sample_size: int) -> Dict[str, Any]:
    """
    非文本类处理器：提取图像的结构化元数据。
    """
    try:
        with Image.open(file_path) as img:
            return {
                "type": "structured",
                "data": {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                }
            }
    except Exception as e:
        return {"error": f"无法读取图像文件: {e}"}


def _read_csv_metadata(file_path: str, sample_size: int) -> Dict[str, Any]:
    """
    文本类处理器：提取CSV文件的原始文本样本。
    """
    try:
        df = pd.read_csv(file_path, nrows=sample_size)
        return {
            "type": "raw_sample",
            "content": df.to_csv(index=False)
        }
    except Exception as e:
        return {"error": f"无法读取CSV文件: {e}"}


def _read_json_metadata(file_path: str, sample_size: int) -> Dict[str, Any]:
    """
    文本类处理器：提取JSON文件的原始文本样本。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 如果是列表，进行采样
        if isinstance(data, list):
            sample_data = data[:sample_size]
        else:
            sample_data = data

        return {
            "type": "raw_sample",
            "content": json.dumps(sample_data, ensure_ascii=False, indent=2)
        }
    except Exception as e:
        return {"error": f"无法读取JSON文件: {e}"}


def _read_jsonl_metadata(file_path: str, sample_size: int) -> Dict[str, Any]:
    """
    文本类处理器：提取JSONL文件的原始文本样本。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [next(f) for _ in range(sample_size)]
        return {
            "type": "raw_sample",
            "content": "".join(lines)
        }
    except StopIteration:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {"type": "raw_sample", "content": f.read()}
    except Exception as e:
        return {"error": f"无法读取JSONL文件: {e}"}


def _read_text_metadata(file_path: str, sample_size: int) -> Dict[str, Any]:
    """
    文本类处理器：提取纯文本文件的原始文本样本。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [next(f) for _ in range(sample_size)]
        return {
            "type": "raw_sample",
            "content": "".join(lines)
        }
    except StopIteration:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {"type": "raw_sample", "content": f.read()}
    except Exception as e:
        return {"error": f"无法读取文本文件: {e}"}


def _read_parquet_metadata(file_path: str, sample_size: int) -> Dict[str, Any]:
    """
    文本类处理器：提取Parquet文件的原始文本样本。
    """
    try:
        df = pd.read_parquet(file_path)
        sample_df = df.head(sample_size)
        return {
            "type": "raw_sample",
            "content": sample_df.to_string()
        }
    except ImportError:
        return {"error": "无法读取Parquet文件: 请安装 'pyarrow' 或 'fastparquet' 库。"}
    except Exception as e:
        return {"error": f"无法读取Parquet文件: {e}"}


# --- Main Dispatcher Function ---

METADATA_READERS = {
    # 非文本类
    '.jpg': _read_image_metadata,
    '.jpeg': _read_image_metadata,
    '.png': _read_image_metadata,
    '.bmp': _read_image_metadata,
    '.gif': _read_image_metadata,
    # 文本类
    '.csv': _read_csv_metadata,
    '.json': _read_json_metadata,
    '.jsonl': _read_jsonl_metadata,
    '.txt': _read_text_metadata,
    '.parquet': _read_parquet_metadata,
}


def read_file_metadata(file_path: str, sample_size: int = 5) -> Dict[str, Any]:
    """
    根据文件类型分派任务，读取并返回文件的元数据。
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return {"error": "文件不存在或不是一个有效的文件。"}

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    reader_func = METADATA_READERS.get(ext)

    if reader_func:
        logging.info(f"  [MetaRead] 使用 '{reader_func.__name__}' 读取 '{os.path.basename(file_path)}'")
        return reader_func(file_path, sample_size=sample_size)
    else:
        return {"status": "unsupported_type", "extension": ext}


if __name__ == "__main__":
    # 测试代码
    test_file = r"D:\Code\datapter\data\output\input_dataset\opus-100\af-en\train-00000-of-00001.parquet"
    result = read_file_metadata(test_file, sample_size=5)
    logging.info(json.dumps(result, indent=2, ensure_ascii=False))
