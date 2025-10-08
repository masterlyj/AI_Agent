import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

def get_llm():
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        print("未检测到 DEEPSEEK_API_KEY，将使用 Google Gemini 模型。")
        return init_chat_model(
            model="gemini-2.5-flash", # Google 的模型名称
            model_provider="google_genai",
            api_key=google_api_key,
            temperature=0,
        )

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        print("检测到 DEEPSEEK_API_KEY，将使用 DeepSeek 模型。")
        # 注意：对于DeepSeek这类兼容OpenAI接口的模型，
        # 通常需要提供一个 base_url。请根据你的 `init_chat_model` 函数进行调整。
        return init_chat_model(
            model="deepseek-chat",  # DeepSeek 的模型名称，例如 'deepseek-chat'
            model_provider="deepseek", # 假设你的函数需要这个参数来区分服务商
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1", # DeepSeek 官方的 API 地址
            temperature=0,
        )

    # 3. 如果两个 Key 都没有，则抛出异常
    raise ValueError(
        "未检测到 DEEPSEEK_API_KEY 或 GOOGLE_API_KEY 环境变量，"
        "请在 .env 文件中至少配置一个。"
    )
