from typing import Any, Dict
import asyncio

from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr


class DeepSeekFactory:
    """DeepSeek模型工厂类，用于创建基于DeepSeek API的LLM实例"""

    @staticmethod
    def create_llm(
            config: Dict[str, Any],
            temperature: float,
            streaming: bool,
            **kwargs
    ) -> Any:
        """创建基于DeepSeek API的LLM实例

        Args:
            config (Dict[str, Any]): 模型配置
            temperature (float): 温度参数
            streaming (bool): 是否使用流式输出
            **kwargs: 其他参数

        Returns:
            Any: LLM实例
        """
        return ChatDeepSeek(
            temperature=temperature,
            model=config['model_name'],
            api_key=SecretStr(config["api_key"]),
        )

    @staticmethod
    async def async_create_llm(
            config: Dict[str, Any],
            temperature: float,
            streaming: bool,
            **kwargs
    ) -> Any:
        """异步创建基于OpenAI API的LLM实例

        Args:
            config (Dict[str, Any]): 模型配置
            temperature (float): 温度参数
            streaming (bool): 是否使用流式输出
            **kwargs: 其他参数

        Returns:
            Any: LLM实例
        """
        # 由于ChatOpenAI的创建是同步的，我们使用run_in_executor来避免阻塞
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: DeepSeekFactory.create_llm(config, temperature, streaming, **kwargs)
        )