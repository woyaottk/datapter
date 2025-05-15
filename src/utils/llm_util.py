import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.utils.SnowFlake import Snowflake

load_dotenv()

sf = Snowflake(worker_id=0, datacenter_id=0)

# 添加LLM实例缓存
_llm_cache = {}
_llm_cache_lock = asyncio.Lock()


def create_llm(
    model_name: str,
    api_key: str,
    api_base: str,
    temperature: float,
    streaming: bool = False,
):
    """创建LLM实例

    Args:
        model_name (str): 模型名称
        api_key (str): API密钥
        api_base (str): API基础URL
        streaming (bool): 是否启用流式输出
    """
    # 创建缓存键
    cache_key = f"{model_name}_{temperature}_{streaming}"

    # 检查缓存中是否已存在相同配置的LLM实例
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    try:
        llm = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            openai_api_key=SecretStr(api_key),
            openai_api_base=api_base,
            verbose=True,
            streaming=streaming,
            request_timeout=600,
            max_retries=3,
        )

        # 限制缓存大小
        if len(_llm_cache) > 10:  # 最多缓存10个不同配置的LLM
            # 移除最早创建的实例
            oldest_key = next(iter(_llm_cache))
            del _llm_cache[oldest_key]

        # 将新创建的LLM实例添加到缓存
        _llm_cache[cache_key] = llm
        return llm
    except Exception as e:
        print(f"Error creating LLM: {str(e)}", exc_info=True)
        raise


async def async_create_llm(
    model_name: str = "glm-4-flash",
    api_key: str = "8781e1f71bb2d17b9408bf56d82bf787.agAcRxyobukflHH8",
    api_base: str = "https://open.bigmodel.cn/api/paas/v4/",
    temperature: float = 0.7,
    streaming: bool = False,
):
    """创建或复用LLM实例"""
    cache_key = f"{model_name}_{temperature}_{streaming}"

    async with _llm_cache_lock:
        if cache_key in _llm_cache:
            return _llm_cache[cache_key]

        try:
            llm = ChatOpenAI(
                temperature=temperature,
                model_name=model_name,
                openai_api_key=SecretStr(api_key),
                openai_api_base=api_base,
                verbose=True,
                streaming=streaming,  # 暂时关闭streaming以便调试
                request_timeout=600,  # 增加超时时间
                max_retries=3,  # 添加重试次数
            )
            # 限制缓存大小
            if len(_llm_cache) > 10:  # 最多缓存10个不同配置的LLM
                # 移除最早创建的实例
                oldest_key = next(iter(_llm_cache))
                del _llm_cache[oldest_key]

            _llm_cache[cache_key] = llm
            return llm
        except Exception as e:
            print(f"Error creating LLM: {str(e)}", exc_info=True)
            raise
