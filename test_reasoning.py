#!/usr/bin/env python3
"""
测试思考推理过程的简单脚本
运行此脚本可以看到思考过程是否正常生成
"""
import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from src.Knowledge_Graph_Agent.agent import RAGAgent

async def test_reasoning():
    print("=" * 60)
    print("🧪 测试思考推理过程")
    print("=" * 60)
    
    # 创建agent
    print("\n1️⃣ 初始化 RAG Agent...")
    agent = await RAGAgent.create(
        working_dir="data/rag_storage",
        storage_mode="memory"  # 使用内存模式快速测试
    )
    print("✅ Agent 初始化完成")
    
    # 测试查询
    test_question = "什么是保险豁免?"
    print(f"\n2️⃣ 测试查询: {test_question}")
    print("-" * 60)
    
    reasoning_displayed = False
    answer_displayed = False
    
    async for chunk in agent.query_stream(
        question=test_question,
        mode="hybrid",
        enable_rerank=False
    ):
        chunk_type = chunk.get("type")
        
        if chunk_type == "status":
            print(f"📊 状态: {chunk.get('content')}")
        
        elif chunk_type == "context":
            print(f"📚 上下文已加载")
        
        elif chunk_type == "reasoning_chunk":
            content = chunk.get("content", "")
            is_done = chunk.get("done", False)
            
            if not reasoning_displayed:
                print("\n🧠 思考推理过程:")
                print("-" * 60)
                reasoning_displayed = True
            
            if content:
                print(content, end="", flush=True)
            
            if is_done:
                print("\n" + "-" * 60)
                print(f"✅ 思考完成 (共 {len(chunk.get('full_reasoning', ''))} 字符)")
        
        elif chunk_type == "answer_chunk":
            content = chunk.get("content", "")
            is_done = chunk.get("done", False)
            
            if not answer_displayed:
                print("\n💬 最终答案:")
                print("-" * 60)
                answer_displayed = True
            
            if content:
                print(content, end="", flush=True)
            
            if is_done:
                print("\n" + "-" * 60)
                print(f"✅ 答案完成 (共 {len(chunk.get('full_answer', ''))} 字符)")
        
        elif chunk_type == "complete":
            print("\n✅ 查询完成!")
        
        elif chunk_type == "error":
            print(f"\n❌ 错误: {chunk.get('content')}")
    
    print("\n" + "=" * 60)
    print("🎉 测试完成")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_reasoning())

