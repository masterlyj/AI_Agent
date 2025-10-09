"""
LightRAG 核心模块 - 知识图谱构建和检索
集成到 LangGraph 工作流中
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
import networkx as nx
import json
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """实体数据结构""" 
    id: str
    name: str
    type: str
    description: str
    properties: Dict[str, Any]

@dataclass
class Relation:
    """关系数据结构"""
    source: str
    target: str
    relation_type: str
    description: str
    confidence: float

class LightRAGKnowledgeGraph:
    """LightRAG 知识图谱核心类"""
    
    def __init__(self, embedder: Embeddings):
        self.embedder = embedder
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.entity_embeddings = {}
        self.entity_vectorstore = None
        
    def extract_entities(self, documents: List[Document]) -> List[Entity]:
        """从文档中提取实体"""
        entities = []
        entity_counter = 0
        
        for doc in documents:
            # 简单的实体提取逻辑 - 可以替换为更复杂的NLP模型
            text = doc.page_content
            words = text.split()
            
            # 提取可能的实体（名词短语、专业术语等）
            potential_entities = self._extract_potential_entities(text)
            
            for entity_text in potential_entities:
                if len(entity_text) > 2 and entity_text not in [e.name for e in entities]:
                    entity = Entity(
                        id=f"entity_{entity_counter}",
                        name=entity_text,
                        type=self._classify_entity_type(entity_text),
                        description=self._generate_entity_description(entity_text, text),
                        properties={"source_doc": doc.metadata.get("source", "unknown")}
                    )
                    entities.append(entity)
                    entity_counter += 1
                    
        return entities
    
    def extract_relations(self, documents: List[Document], entities: List[Entity]) -> List[Relation]:
        """从文档中提取实体间关系"""
        relations = []
        entity_names = {e.name: e for e in entities}
        
        for doc in documents:
            text = doc.page_content
            # 简单的共现关系提取
            sentences = text.split('.')
            
            for sentence in sentences:
                sentence_entities = [e for e in entities if e.name.lower() in sentence.lower()]
                
                # 如果句子中包含多个实体，创建关系
                if len(sentence_entities) >= 2:
                    for i in range(len(sentence_entities)):
                        for j in range(i + 1, len(sentence_entities)):
                            relation = Relation(
                                source=sentence_entities[i].id,
                                target=sentence_entities[j].id,
                                relation_type=self._extract_relation_type(sentence),
                                description=sentence.strip(),
                                confidence=0.8  # 简单置信度计算
                            )
                            relations.append(relation)
                            
        return relations
    
    def build_graph(self, documents: List[Document]) -> Dict[str, Any]:
        """构建知识图谱"""
        logger.info("开始构建知识图谱...")
        
        # 1. 提取实体
        entities = self.extract_entities(documents)
        logger.info(f"提取到 {len(entities)} 个实体")
        
        # 2. 提取关系
        relations = self.extract_relations(documents, entities)
        logger.info(f"提取到 {len(relations)} 个关系")
        
        # 3. 构建图结构
        for entity in entities:
            self.entities[entity.id] = entity
            self.graph.add_node(
                entity.id, 
                name=entity.name, 
                type=entity.type,
                description=entity.description,
                properties=entity.properties
            )
        
        for relation in relations:
            self.relations.append(relation)
            self.graph.add_edge(
                relation.source,
                relation.target,
                relation_type=relation.relation_type,
                description=relation.description,
                confidence=relation.confidence
            )
        
        # 4. 构建实体向量库
        self._build_entity_vectorstore()
        
        logger.info(f"知识图谱构建完成: {self.graph.number_of_nodes()} 个节点, {self.graph.number_of_edges()} 条边")
        
        return {
            "entities": entities,
            "relations": relations,
            "graph_stats": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges()
            }
        }
    
    def graph_enhanced_retrieve(self, query: str, k: int = 5) -> List[Document]:
        """基于知识图谱的增强检索"""
        # 1. 实体识别
        query_entities = self._identify_query_entities(query)
        
        # 2. 图遍历获取相关实体
        relevant_entities = self._get_relevant_entities(query_entities)
        
        # 3. 基于实体检索相关文档片段
        retrieved_docs = []
        for entity in relevant_entities:
            if entity.id in self.entity_vectorstore.index_to_docstore_id:
                # 从实体向量库检索
                docs = self.entity_vectorstore.similarity_search(
                    query, k=k//len(relevant_entities) + 1
                )
                retrieved_docs.extend(docs)
        
        # 4. 去重和排序
        unique_docs = []
        seen_content = set()
        for doc in retrieved_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        return unique_docs[:k]
    
    def get_graph_summary(self) -> str:
        """获取图谱摘要信息"""
        if not self.graph.nodes():
            return "知识图谱为空"
        
        # 统计信息
        node_types = defaultdict(int)
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            node_types[node_data.get('type', 'unknown')] += 1
        
        summary = f"知识图谱包含 {self.graph.number_of_nodes()} 个实体和 {self.graph.number_of_edges()} 个关系:\n"
        for entity_type, count in node_types.items():
            summary += f"- {entity_type}: {count} 个\n"
        
        return summary
    
    def _extract_potential_entities(self, text: str) -> List[str]:
        """提取潜在实体 - 简化版本"""
        # 这里可以使用更复杂的NLP模型，如spaCy, NLTK等
        import re
        
        # 提取大写字母开头的单词序列
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
        
        # 提取引号中的内容
        quoted = re.findall(r'"([^"]*)"', text)
        
        # 提取技术术语（包含数字、连字符等）
        technical = re.findall(r'\b[A-Za-z]+[-_][A-Za-z0-9]+\b', text)
        
        # 提取长名词短语
        noun_phrases = re.findall(r'\b[A-Za-z]+(?:\s+[A-Za-z]+){2,}\b', text)
        
        entities = list(set(capitalized + quoted + technical + noun_phrases))
        return [e for e in entities if len(e) > 2 and len(e) < 50]
    
    def _classify_entity_type(self, entity_text: str) -> str:
        """简单的实体类型分类"""
        entity_lower = entity_text.lower()
        
        if any(word in entity_lower for word in ['method', 'algorithm', 'model', 'approach']):
            return "METHOD"
        elif any(word in entity_lower for word in ['dataset', 'data', 'corpus']):
            return "DATASET"
        elif any(word in entity_lower for word in ['metric', 'measure', 'score']):
            return "METRIC"
        elif any(word in entity_lower for word in ['task', 'problem', 'challenge']):
            return "TASK"
        else:
            return "CONCEPT"
    
    def _generate_entity_description(self, entity: str, context: str) -> str:
        """生成实体描述"""
        # 找到包含实体的句子作为描述
        sentences = context.split('.')
        for sentence in sentences:
            if entity.lower() in sentence.lower():
                return sentence.strip()
        return f"实体: {entity}"
    
    def _extract_relation_type(self, sentence: str) -> str:
        """提取关系类型"""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['improves', 'enhances', 'better']):
            return "IMPROVES"
        elif any(word in sentence_lower for word in ['compares', 'compared', 'versus']):
            return "COMPARES"
        elif any(word in sentence_lower for word in ['uses', 'applies', 'employs']):
            return "USES"
        elif any(word in sentence_lower for word in ['based on', 'extends', 'builds']):
            return "BASED_ON"
        else:
            return "RELATED_TO"
    
    def _identify_query_entities(self, query: str) -> List[Entity]:
        """识别查询中的实体"""
        query_entities = []
        query_lower = query.lower()
        
        for entity in self.entities.values():
            if entity.name.lower() in query_lower:
                query_entities.append(entity)
        
        return query_entities
    
    def _get_relevant_entities(self, query_entities: List[Entity], max_depth: int = 2) -> List[Entity]:
        """通过图遍历获取相关实体"""
        relevant_entities = set(query_entities)
        
        for entity in query_entities:
            # 获取直接邻居
            neighbors = list(self.graph.neighbors(entity.id))
            for neighbor_id in neighbors:
                if neighbor_id in self.entities:
                    relevant_entities.add(self.entities[neighbor_id])
        
        return list(relevant_entities)
    
    def _build_entity_vectorstore(self):
        """构建实体向量库"""
        if not self.entities:
            return
        
        entity_texts = []
        entity_metadatas = []
        
        for entity in self.entities.values():
            entity_text = f"{entity.name}: {entity.description}"
            entity_texts.append(entity_text)
            entity_metadatas.append({
                "entity_id": entity.id,
                "entity_name": entity.name,
                "entity_type": entity.type,
                "source": "knowledge_graph"
            })
        
        self.entity_vectorstore = FAISS.from_texts(
            entity_texts, 
            self.embedder,
            metadatas=entity_metadatas
        )
        
        logger.info(f"实体向量库构建完成，包含 {len(entity_texts)} 个实体")

