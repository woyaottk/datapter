import os
from typing import List, Any
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv('.env')

class EmbeddingUtil:
    """Embedding工具类，用于创建和管理Embedding实例"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingUtil, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化Embedding实例"""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
            
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文本转换为向量
        
        Args:
            texts (List[str]): 要转换的文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """将单个文本转换为向量
        
        Args:
            text (str): 要转换的文本
            
        Returns:
            List[float]: 向量
        """
        return self.embeddings.embed_query(text)


# 使用示例
if __name__ == "__main__":
    # 创建 EmbeddingUtil 实例
    embedding_util = EmbeddingUtil()
    
    # 转换单个文本
    vector = embedding_util.embed_query("Hello, world!")
    print(f"Vector length: {len(vector)}")
    
    # 转换多个文本
    vectors = embedding_util.embed_documents(["Hello", "world"])
    print(f"Number of vectors: {len(vectors)}") 