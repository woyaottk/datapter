import logging
from typing import Any, Dict, Type
import asyncio
from dotenv import load_dotenv

from src.llm.factory.deepseek_factory import DeepSeekFactory
from src.llm.factory.openai_factory import OpenAIFactory
from src.llm.factory.tongyi_factory import TongyiFactory
from src.llm.model.LLMType import LLMType

load_dotenv('.env')


class LLMFactory:
    """LLM工厂类，用于创建不同类型的LLM实例"""

    # LLM实例缓存
    _llm_cache: Dict[str, Any] = {}
    _max_cache_size = 10  # 最大缓存实例数
    _cache_lock = asyncio.Lock()  # 异步锁，用于保护缓存访问
    
    # 工厂类映射
    _factory_map: Dict[LLMType, Type] = {
        LLMType.QWEN: OpenAIFactory,
        LLMType.QWEN3: OpenAIFactory,
        LLMType.QWEN_PLUS: OpenAIFactory,
        LLMType.QWEN_MAX: TongyiFactory,
        LLMType.DEEPSEEK_CHAT: DeepSeekFactory,
        LLMType.DEEPSEEK_REASON: DeepSeekFactory,
    }

    @classmethod
    def _get_cache_key(cls, llm_type: LLMType, temperature: float, streaming: bool) -> str:
        """生成缓存键

        Args:
            llm_type (LLMType): LLM类型
            temperature (float): 温度参数
            streaming (bool): 是否使用流式输出

        Returns:
            str: 缓存键
        """
        return f"{llm_type.value}_{temperature}_{streaming}"

    @classmethod
    def _manage_cache_size(cls) -> None:
        """管理缓存大小，如果超过最大限制则移除最早的实例"""
        if len(cls._llm_cache) > cls._max_cache_size:
            # 移除最早创建的实例
            oldest_key = next(iter(cls._llm_cache))
            del cls._llm_cache[oldest_key]

    @classmethod
    def create_llm(
            cls,
            llm_type: LLMType,
            temperature: float = 0.7,
            streaming: bool = False,
            **kwargs
    ) -> Any:
        """创建LLM实例

        Args:
            llm_type (LLMType): LLM类型枚举值
            temperature (float): 温度参数，默认0.7
            streaming (bool): 是否使用流式输出，默认False
            **kwargs: 其他参数

        Returns:
            Any: LLM实例

        Raises:
            ValueError: 当传入不支持的LLM类型时抛出
        """
        # 生成缓存键
        cache_key = cls._get_cache_key(llm_type, temperature, streaming)

        # 检查缓存中是否已存在相同配置的实例
        if cache_key in cls._llm_cache:
            return cls._llm_cache[cache_key]

        try:
            # 获取模型配置
            model_config = LLMType.get_config(llm_type)

            # 创建新的LLM实例
            llm = cls._create_llm_by_type(model_config, temperature, streaming, **kwargs)

            # 管理缓存大小
            cls._manage_cache_size()

            # 将新创建的实例添加到缓存
            cls._llm_cache[cache_key] = llm

            return llm

        except Exception as e:
            raise ValueError(f"Failed to create LLM instance for {llm_type}: {str(e)}")

    @classmethod
    async def async_create_llm(
            cls,
            llm_type: LLMType,
            temperature: float = 0.7,
            streaming: bool = False,
            **kwargs
    ) -> Any:
        """异步创建LLM实例

        Args:
            llm_type (LLMType): LLM类型枚举值
            temperature (float): 温度参数，默认0.7
            streaming (bool): 是否使用流式输出，默认False
            **kwargs: 其他参数

        Returns:
            Any: LLM实例

        Raises:
            ValueError: 当传入不支持的LLM类型时抛出
        """
        # 生成缓存键
        cache_key = cls._get_cache_key(llm_type, temperature, streaming)

        # 使用异步锁保护缓存访问
        async with cls._cache_lock:
            # 检查缓存中是否已存在相同配置的实例
            if cache_key in cls._llm_cache:
                return cls._llm_cache[cache_key]

            try:
                # 获取模型配置
                model_config = LLMType.get_config(llm_type)

                # 创建新的LLM实例
                llm = await cls._async_create_llm_by_type(model_config, temperature, streaming, **kwargs)

                # 管理缓存大小
                cls._manage_cache_size()

                # 将新创建的实例添加到缓存
                cls._llm_cache[cache_key] = llm

                return llm

            except Exception as e:
                raise ValueError(f"Failed to create LLM instance for {llm_type}: {str(e)}")

    @classmethod
    def _create_llm_by_type(
            cls,
            config: Dict[str, Any],
            temperature: float,
            streaming: bool,
            **kwargs
    ) -> Any:
        """根据类型创建LLM实例

        Args:
            config (Dict[str, Any]): 模型配置
            temperature (float): 温度参数
            streaming (bool): 是否使用流式输出
            **kwargs: 其他参数

        Returns:
            Any: LLM实例

        Raises:
            ValueError: 当传入不支持的LLM类型时抛出
        """
        try:
            factory_class = cls._factory_map.get(config["type"])
            if not factory_class:
                raise ValueError(f"Unsupported LLM type: {config['type']}")
            
            return factory_class.create_llm(config, temperature, streaming, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create LLM instance for {config['type']}: {str(e)}")

    @classmethod
    async def _async_create_llm_by_type(
            cls,
            config: Dict[str, Any],
            temperature: float,
            streaming: bool,
            **kwargs
    ) -> Any:
        """异步创建LLM实例

        Args:
            config (Dict[str, Any]): 模型配置
            temperature (float): 温度参数
            streaming (bool): 是否使用流式输出
            **kwargs: 其他参数

        Returns:
            Any: LLM实例

        Raises:
            ValueError: 当传入不支持的LLM类型时抛出
        """
        try:
            factory_class = cls._factory_map.get(config["type"])
            if not factory_class:
                raise ValueError(f"Unsupported LLM type: {config['type']}")
            
            return await factory_class.async_create_llm(config, temperature, streaming, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create LLM instance for {config['type']}: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 同步创建 Qwen 模型实例
    qwen_llm1 = LLMFactory.create_llm(LLMType.QWEN)

    # 异步创建 Qwen 模型实例
    async def main():
        qwen_llm2 = await LLMFactory.async_create_llm(LLMType.QWEN)
        logging.info("Async LLM created successfully")

    asyncio.run(main())