#!/usr/bin/env python3
"""
LightRAG 快速启动脚本
简化版本，适合快速体验 LightRAG 功能
"""

import sys
import os
import subprocess
import time

def check_ollama():
    """检查 Ollama 服务状态"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama 服务正常运行")
            return True
        else:
            print("❌ Ollama 服务未运行")
            return False
    except FileNotFoundError:
        print("❌ Ollama 未安装")
        return False

def check_model():
    """检查嵌入模型是否已下载"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'qwen3-embedding:0.6b' in result.stdout:
            print("✅ 嵌入模型已下载")
            return True
        else:
            print("⚠️ 嵌入模型未下载，正在下载...")
            subprocess.run(['ollama', 'pull', 'qwen3-embedding:0.6b'])
            print("✅ 嵌入模型下载完成")
            return True
    except Exception as e:
        print(f"❌ 模型检查失败: {e}")
        return False

def check_env():
    """检查环境变量"""
    if os.path.exists('.env'):
        print("✅ 环境配置文件存在")
        return True
    else:
        print("⚠️ 未找到 .env 文件，请创建并添加 GOOGLE_API_KEY")
        return False

def run_demo():
    """运行演示"""
    print("\n🚀 启动 LightRAG 演示...")
    try:
        # 导入并运行演示
        sys.path.append(os.path.dirname(__file__))
        from src.Paper_Study_Agent.demo_lightrag import demo
        
        print("📊 对比界面将在 http://127.0.0.1:7860 启动")
        print("💡 提示：在浏览器中打开上述地址体验 LightRAG 功能")
        print("🔍 建议问题：'LightRAG 的核心创新点是什么？'")
        
        demo.launch(server_port=7860, share=False)
        
    except Exception as e:
        print(f"❌ 演示启动失败: {e}")
        print("💡 请检查依赖安装和配置")

def main():
    """主函数"""
    print("🎯 LightRAG 快速启动检查")
    print("=" * 50)
    
    # 检查前置条件
    checks = [
        ("Ollama 服务", check_ollama),
        ("嵌入模型", check_model),
        ("环境配置", check_env),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n🔍 检查 {check_name}...")
        if not check_func():
            all_passed = False
            print(f"❌ {check_name} 检查失败")
        else:
            print(f"✅ {check_name} 检查通过")
    
    if not all_passed:
        print("\n⚠️ 部分检查失败，请解决上述问题后重试")
        print("\n📚 详细说明请参考: LIGHTRAG_DEMO_USAGE.md")
        return
    
    print("\n🎉 所有检查通过！")
    
    # 询问是否启动演示
    try:
        response = input("\n🚀 是否启动 LightRAG 演示？(y/n): ").lower().strip()
        if response in ['y', 'yes', '是', '']:
            run_demo()
        else:
            print("👋 已取消启动")
    except KeyboardInterrupt:
        print("\n👋 已取消启动")

if __name__ == "__main__":
    main()

