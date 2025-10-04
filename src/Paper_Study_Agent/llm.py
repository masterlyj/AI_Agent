import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("未检测到GOOGLE_API_KEY环境变量，请在.env文件中配置。")
    return init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        api_key=api_key,
        temperature=0,
    )