from typing import TypedDict, List, Dict, Any, Optional, Literal

class IndexingState(TypedDict):
    working_dir: str
    inputs: List[str]
    ids: Optional[List[str]]
    file_paths: Optional[List[str]]
    track_id: Optional[str]
    status_message: str

class QueryState(TypedDict):
    working_dir: str
    query: str
    query_mode: Literal["naive", "local", "global", "hybrid"]
    context: Dict[str, Any]
    answer: str