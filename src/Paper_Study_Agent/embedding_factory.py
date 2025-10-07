from typing import List
from langchain_core.embeddings import Embeddings as BaseEmbeddings

def get_embedder(embedding_config: dict) -> BaseEmbeddings:
    etype = embedding_config["type"]

    if etype == "hf":
        model_name = embedding_config.get("model_name")
        if not model_name:
            raise ValueError("缺少 HuggingFace 嵌入模型的 'model_name'")
        
        # 高级参数
        model_kwargs = embedding_config.get("model_kwargs", {})  # 传递给模型初始化的参数，如 device、trust_remote_code 等
        encode_kwargs = embedding_config.get("encode_kwargs", {})  # 文档编码时的参数，如 batch_size、normalize_embeddings 等
        show_progress = embedding_config.get("show_progress", False)  # 是否显示进度条
        multi_process = embedding_config.get("multi_process", False)  # 是否启用多进程加速

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
            # 你可以加更多参数
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
            api_key=embedding_config.get("api_key", None),
        )

    else:
        raise ValueError(f"不支持的嵌入类型: {etype}。支持类型: hf, vllm, ollama")