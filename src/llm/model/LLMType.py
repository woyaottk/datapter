from enum import Enum
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv('.env')

# 配置缓存
_config_cache: Dict['LLMType', Dict[str, Any]] = {}
_config_initialized = False

class LLMType(Enum):
    """LLM类型枚举"""
    QWEN = "QWEN"
    QWEN_MAX = "QWEN_MAX"
    DEEPSEEK_CHAT = "DEEPSEEK_CHAT"
    DEEPSEEK_REASON = "DEEPSEEK_REASON"

    @classmethod
    def _initialize_config_cache(cls) -> None:
        """初始化配置缓存，加载所有模型的配置信息"""
        global _config_initialized, _config_cache
        if _config_initialized:
            return
            
        for llm_type in cls:
            _config_cache[llm_type] = {
                "type": llm_type,
                "model_name": os.getenv(f"{llm_type.name}.MODEL_NAME"),
                "base_url": os.getenv(f"{llm_type.name}.BASE_URL"),
                "api_key": os.getenv(f"{llm_type.name}.API_KEY")
            }
        
        _config_initialized = True

    @classmethod
    def get_config(cls, llm_type: 'LLMType') -> Dict[str, Any]:
        """获取模型配置
        
        Args:
            llm_type (LLMType): LLM类型枚举值
            
        Returns:
            Dict[str, Any]: 模型配置字典
            
        Raises:
            ValueError: 当模型类型不支持或配置缺失时抛出
        """
        global _config_cache
        # 确保配置缓存已初始化
        cls._initialize_config_cache()
        
        config = _config_cache.get(llm_type)
        if not config:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
            
        # 检查必要的配置是否存在
        if not config["api_key"]:
            raise ValueError(f"API key not found for {llm_type.value}")
            
        return config