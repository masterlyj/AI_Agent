from typing import Dict, List, Any, Annotated, TypedDict, Optional
from langchain_core.documents import Document
from langchain_core.language_models import FakeStreamingListLLM
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

class QueryParam(BaseModel):
    """当前未使用，保留扩展性"""
    pass

class Paper_Study_State(TypedDict):
    """The state of a paper study."""

    thread_id: str
    arXiv_ids: List[str] # The arXiv of the papers.
    query: QueryParam
    context: List[Document]
    answer: str

    embedder: Embeddings
    vectorstore: FAISS
    convstore: FAISS

    context_retrieved: str
    history_retrieved: str
