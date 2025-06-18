from typing import Any, Dict
import asyncio
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

class OpenAIFactory:
    """OpenAI模型工厂类，用于创建基于OpenAI API的LLM实例"""
    
    @staticmethod
    def create_llm(
        config: Dict[str, Any],
        temperature: float,
        streaming: bool,
        **kwargs
    ) -> Any:
        """创建基于OpenAI API的LLM实例
        
        Args:
            config (Dict[str, Any]): 模型配置
            temperature (float): 温度参数
            streaming (bool): 是否使用流式输出
            **kwargs: 其他参数
            
        Returns:
            Any: LLM实例
        """
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
            lambda: OpenAIFactory.create_llm(config, temperature, streaming, **kwargs)
        ) 