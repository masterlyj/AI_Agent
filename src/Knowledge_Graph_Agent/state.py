from typing import TypedDict, List, Dict, Any, Optional, Literal
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

class IndexingState(TypedDict):
    working_dir: str
    inputs: List[str]
    ids: Optional[List[str]]
    file_paths: Optional[List[str]]
    track_id: Optional[str]
    status_message: str

class QueryState(TypedDict):
    thread_id: str
    working_dir: str
    query: str
    llm: BaseChatModel
    query_mode: Literal["naive", "local", "global", "hybrid", "mix"]
    reranker: Optional[Any]
    rerank_top_k: int
    retrieved_docs: Optional[List[Document]]
    retrieved_entities: Optional[List[Dict[str, Any]]]
    retrieved_relationships: Optional[List[Dict[str, Any]]]
    final_docs: Optional[List[Document]]
    context: Dict[str, Any]
    answer: str
    chat_history: Optional[List[Dict[str, str]]]