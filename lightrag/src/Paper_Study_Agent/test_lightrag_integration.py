"""
LightRAG 集成测试脚本
验证 LightRAG 与 LangGraph 的集成是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Paper_Study_Agent.app import PaperChatBot
from Paper_Study_Agent.embedding_factory import get_embedder
from Paper_Study_Agent.lightrag_core import LightRAGKnowledgeGraph
from langchain_core.documents import Document

def test_lightrag_core():
    """测试 LightRAG 核心功能"""
    print("🧪 测试 LightRAG 核心功能...")
    
    # 创建测试嵌入器
    embedding_config = {
        "type": "ollama",
        "model": "qwen3-embedding:0.6b"
    }
    
    try:
        embedder = get_embedder(embedding_config)
        print("✅ 嵌入器创建成功")
    except Exception as e:
        print(f"❌ 嵌入器创建失败: {e}")
        return False
    
    # 创建测试文档
    test_docs = [
        Document(
            page_content="LightRAG is a simple and fast retrieval-augmented generation framework that uses knowledge graphs to improve RAG performance.",
            metadata={"source": "test_doc_1"}
        ),
        Document(
            page_content="The main innovation of LightRAG is the graph-enhanced retrieval strategy that leverages entity relationships for better context understanding.",
            metadata={"source": "test_doc_2"}
        )
    ]
    
    # 测试知识图谱构建
    try:
        kg = LightRAGKnowledgeGraph(embedder)
        result = kg.build_graph(test_docs)
        print(f"✅ 知识图谱构建成功: {result['graph_stats']}")
        
        # 测试图增强检索
        test_query = "What is the main innovation of LightRAG?"
        retrieved_docs = kg.graph_enhanced_retrieve(test_query, k=2)
        print(f"✅ 图增强检索成功，检索到 {len(retrieved_docs)} 个文档")
        
        # 测试图谱摘要
        summary = kg.get_graph_summary()
        print(f"✅ 图谱摘要生成成功: {summary[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 知识图谱测试失败: {e}")
        return False

def test_paper_chatbot():
    """测试完整的 PaperChatBot 集成"""
    print("\n🧪 测试 PaperChatBot 集成...")
    
    # 配置
    arxiv_ids = ["2410.05779"]  # LightRAG 论文
    embedding_config = {
        "type": "ollama", 
        "model": "qwen3-embedding:0.6b"
    }
    
    try:
        # 测试传统 RAG
        print("📚 测试传统 RAG...")
        bot_traditional = PaperChatBot(
            arxiv_ids=arxiv_ids,
            embedding_config=embedding_config,
            use_lightrag=False
        )
        print("✅ 传统 RAG 机器人创建成功")
        
        # 测试 LightRAG
        print("🧠 测试 LightRAG...")
        bot_lightrag = PaperChatBot(
            arxiv_ids=arxiv_ids,
            embedding_config=embedding_config,
            use_lightrag=True
        )
        print("✅ LightRAG 机器人创建成功")
        
        # 测试对话功能
        test_message = "What is LightRAG?"
        print(f"\n💬 测试问题: {test_message}")
        
        # 传统 RAG 回答
        traditional_response = bot_traditional.chat(test_message, [])
        print(f"📚 传统 RAG 回答: {traditional_response[:100]}...")
        
        # LightRAG 回答
        lightrag_response = bot_lightrag.chat(test_message, [])
        print(f"🧠 LightRAG 回答: {lightrag_response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ PaperChatBot 测试失败: {e}")
        return False

def test_state_management():
    """测试状态管理"""
    print("\n🧪 测试状态管理...")
    
    from Paper_Study_Agent.state import Paper_Study_State
    
    try:
        # 创建测试状态
        test_state: Paper_Study_State = {
            "thread_id": "test",
            "arXiv_ids": ["2410.05779"],
            "query": "test query",
            "context": [],
            "answer": "",
            "embedder": None,
            "vectorstore": None,
            "convstore": None,
            "knowledge_graph": None,
            "graph_context": "",
            "use_lightrag": True,
            "context_retrieved": "",
            "history_retrieved": "",
            "messages": [],
        }
        
        print("✅ 状态创建成功")
        print(f"   - 使用 LightRAG: {test_state['use_lightrag']}")
        print(f"   - 线程 ID: {test_state['thread_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 状态管理测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🚀 开始 LightRAG 集成测试...\n")
    
    tests = [
        ("LightRAG 核心功能", test_lightrag_core),
        ("PaperChatBot 集成", test_paper_chatbot),
        ("状态管理", test_state_management),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"🧪 运行测试: {test_name}")
        print(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
        
        print()
    
    print(f"{'='*50}")
    print(f"📊 测试结果: {passed}/{total} 通过")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 所有测试通过！LightRAG 集成成功！")
        return True
    else:
        print("⚠️ 部分测试失败，请检查配置和依赖")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

