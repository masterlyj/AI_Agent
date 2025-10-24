from typing import TypedDict, List, Dict, Any, Optional, Literal
#新增Document导入
from langchain_core.documents import Document

class IndexingState(TypedDict):
    working_dir: str
    inputs: List[str]
    ids: Optional[List[str]]
    file_paths: Optional[List[str]]
    track_id: Optional[str]
    status_message: str

class QueryState(TypedDict):
    thread_id: int
    working_dir: str
    query: str
    query_mode: Literal["naive", "local", "global", "hybrid"]
    reranker: Optional[Any]  # RerankerModel 实例或 None
    #检索到的文档列表 (粗排结果)
    retrieved_docs: Optional[List[Document]]
    # 新增：检索到的实体和关系
    retrieved_entities: Optional[List[Dict[str, Any]]]
    retrieved_relationships: Optional[List[Dict[str, Any]]]
    #精排后的最终文档列表
    final_docs: Optional[List[Document]]
    context: Dict[str, Any]
    answer: str