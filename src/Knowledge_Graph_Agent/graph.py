# src/Knowledge_Graph_Agent/graph.py

from langgraph.graph import StateGraph, END
from .state import IndexingState, QueryState

def create_indexing_graph(nodes):
    workflow = StateGraph(IndexingState)
    workflow.add_node("index_documents", nodes.index_documents)
    workflow.set_entry_point("index_documents")
    workflow.add_edge("index_documents", END)
    return workflow.compile()

def create_querying_graph(nodes):
    workflow = StateGraph(QueryState)
    workflow.add_node("retrieve_context", nodes.retrieve_context)
    workflow.add_node("generate_answer", nodes.generate_answer)
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", END)
    return workflow.compile()