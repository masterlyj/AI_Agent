from typing import List
from langchain_core.embeddings import Embeddings as BaseEmbeddings

def get_embedder(embedding_config: dict) -> BaseEmbeddings:
    """
    根据配置创建嵌入模型实例
    
    Args:
        embedding_config: 嵌入模型配置字典，包含以下字段：
            - type: 嵌入模型类型 (hf, ollama, vllm)
            - 其他特定于模型类型的参数
    
    Returns:
        BaseEmbeddings: LangChain 嵌入模型实例
    """
    etype = embedding_config.get("type")
    
    if not etype:
        raise ValueError("embedding_config 必须包含 'type' 字段")

    if etype == "hf":
        model_name = embedding_config.get("model_name")
        if not model_name:
            raise ValueError("缺少 HuggingFace 嵌入模型的 'model_name'")
        
        # 高级参数
        model_kwargs = embedding_config.get("model_kwargs", {})
        encode_kwargs = embedding_config.get("encode_kwargs", {})
        show_progress = embedding_config.get("show_progress", False)
        multi_process = embedding_config.get("multi_process", False)

        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=show_progress,
            multi_process=multi_process,
        )
    
    elif etype == "ollama":
        from langchain_ollama.embeddings import OllamaEmbeddings
        model = embedding_config.get("model")
        if model is None:
            raise ValueError("缺少 Ollama 嵌入模型的 'model' 参数")
        base_url = embedding_config.get("base_url", "http://localhost:11434")
        return OllamaEmbeddings(
            model=model,
            base_url=base_url,
        )

    elif etype == "vllm":
        from langchain_openai import OpenAIEmbeddings
        model = embedding_config.get("model")
        base_url = embedding_config.get("base_url")
        if model is None or base_url is None:
            raise ValueError("缺少 vllm 嵌入模型的 'model' 或 'base_url' 参数")
        return OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=embedding_config.get("api_key", "EMPTY"),
        )

    else:
        raise ValueError(f"不支持的嵌入类型: {etype}。支持类型: hf, vllm, ollama")


def create_lightrag_embedding_adapter(langchain_embedder: BaseEmbeddings, embedding_dim: int = 1024):
    """
    将 LangChain 嵌入模型适配为 LightRAG 兼容的嵌入函数
    
    Args:
        langchain_embedder: LangChain 嵌入模型实例
        embedding_dim: 嵌入向量维度
    
    Returns:
        EmbeddingFunc: LightRAG 兼容的嵌入函数对象
    """
    # 导入 EmbeddingFunc
    from .utils import EmbeddingFunc

    async def embedding_func(texts: List[str]) -> List[List[float]]:
        """
        LightRAG 兼容的嵌入函数
        
        Args:
            texts: 待嵌入的文本列表
        
        Returns:
            嵌入向量列表
        """
        if not texts:
            return []
        
        # LangChain 嵌入模型通常是同步的，需要在异步环境中运行
        import asyncio
        from asyncio import to_thread
        
        # 使用 to_thread 在后台线程中运行同步嵌入
        embeddings = await to_thread(langchain_embedder.embed_documents, texts)
        return embeddings

    # 返回 EmbeddingFunc 对象
    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=16384,
        func=embedding_func
    )