from typing import List
from langchain_core.embeddings import Embeddings as BaseEmbeddings
from openai import OpenAI

class OllamaEmbeddingsWrapper(BaseEmbeddings):
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", api_key: str = "not-needed"):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 逐个嵌入，避免 Ollama 批量输入不兼容
        return [self.embed_query(text) for text in texts]

# --- 修改 get_embedder ---
def get_embedder(embedding_config: dict) -> BaseEmbeddings:
    etype = embedding_config["type"]

    if etype == "hf":
        if "model_name" not in embedding_config:
            raise ValueError("Missing 'model_name' for HuggingFace embedder")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=embedding_config["model_name"])
    
    elif etype == "vllm":
        if "model" not in embedding_config:
            raise ValueError(f"Missing 'model' for {etype} embedder")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=embedding_config["model"],
            base_url=embedding_config["base_url"],
            api_key=embedding_config.get("api_key", "not-needed")
        )
    
    elif etype == "ollama":
        if "model" not in embedding_config:
            raise ValueError(f"Missing 'model' for {etype} embedder")
        return OllamaEmbeddingsWrapper(
            model=embedding_config["model"],
            base_url=embedding_config.get("base_url", "http://localhost:11434/v1"),
            api_key=embedding_config.get("api_key", "not-needed")
        )
    else:
        raise ValueError(f"Unsupported embedding type: {etype}")