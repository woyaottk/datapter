import asyncio
import os
from enum import Enum
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import Tongyi

from pydantic import SecretStr

from src.utils.SnowFlake import Snowflake

load_dotenv()

sf = Snowflake(worker_id=0, datacenter_id=0)

class LLMType(Enum):
    """LLM类型枚举"""
    OPENAI = "openai"
    QWEN = "qwen"
    QWEN_MAX = "qwen-max"
    GLM = "glm"

# 添加LLM实例缓存
_llm_cache: Dict[str, Any] = {}
_llm_cache_lock = asyncio.Lock()

def get_model_config():
    """从环境变量获取模型配置"""
    model_selection = os.getenv("MODEL_SELECTION", "QWEN-MAX")

    if model_selection == "QWEN":
        type = LLMType.QWEN
    elif  model_selection == "QWEN-MAX":
        type = LLMType.QWEN_MAX
    elif model_selection == "GLM":
        type = LLMType.GLM
    else:
        raise ValueError(f"Unsupported model selection: {model_selection}")

    return {
        "type": type,
        "model_name": os.getenv(f"{model_selection}.MODEL_NAME"),
        "base_url": os.getenv(f"{model_selection}.BASE_URL"),
        "api_key": os.getenv(f"{model_selection}.API_KEY")
    }

def create_llm(temperature: float = 0.7, streaming: bool = False, **kwargs):
    """创建LLM实例

    Args:
        temperature (float): 温度参数
        streaming (bool): 是否使用流式输出
        **kwargs: 其他参数

    Returns:
        Any: LLM实例
    """
    config = get_model_config()
    cache_key = f"{config['type'].value}_{config['model_name']}_{temperature}_{streaming}"

    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    try:
        llm = _create_llm_by_type(config, temperature, streaming, **kwargs)

        if len(_llm_cache) > 10:
            oldest_key = next(iter(_llm_cache))
            del _llm_cache[oldest_key]

        _llm_cache[cache_key] = llm
        return llm
    except Exception as e:
        print(f"Error creating LLM: {str(e)}", exc_info=True)
        raise

def _create_llm_by_type(config: dict, temperature: float, streaming: bool, **kwargs):
    """根据类型创建LLM实例"""
    if config["type"] == LLMType.QWEN:
        print("使用Qwen模型")
        return ChatOpenAI(
            temperature=temperature,
            model_name=config["model_name"],
            openai_api_key=SecretStr(config["api_key"]),
            openai_api_base=config["base_url"],
            verbose=True,
            streaming=streaming,
            request_timeout=600,
            max_retries=3,
            **kwargs
        )
    elif config["type"] == LLMType.QWEN_MAX:
        print("使用Qwen-MAX模型")
        os.environ["DASHSCOPE_API_KEY"] = config["api_key"]
        return Tongyi(
            model=config["model_name"],
            DASHSCOPE_API_KEY=config["api_key"],
            temperature=0.1,
        )
    elif config["type"] == LLMType.GLM:
        print("使用GLM模型")
        return ChatOpenAI(
                temperature=temperature,
                model_name=config['model_name'],
                openai_api_key=SecretStr(config["api_key"]),
                openai_api_base=config["base_url"],
                verbose=True,
                streaming=streaming,  # 暂时关闭streaming以便调试
                request_timeout=600,  # 增加超时时间
                max_retries=3,  # 添加重试次数
            )
    else:
        raise ValueError(f"Unsupported LLM type: {config['type']}")

async def async_create_llm(temperature: float = 0.7, streaming: bool = False, **kwargs):
    """异步创建LLM实例"""
    config = get_model_config()
    cache_key = f"{config['type'].value}_{config['model_name']}_{temperature}_{streaming}"

    async with _llm_cache_lock:
        if cache_key in _llm_cache:
            return _llm_cache[cache_key]

        try:
            llm = _create_llm_by_type(config, temperature, streaming, **kwargs)
            
            if len(_llm_cache) > 10:
                oldest_key = next(iter(_llm_cache))
                del _llm_cache[oldest_key]

            _llm_cache[cache_key] = llm
            return llm
        except Exception as e:
            print(f"Error creating LLM: {str(e)}", exc_info=True)
            raise
