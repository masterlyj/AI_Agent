import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

def get_llm():
    """
    从 .env 环境变量读取 LLM 配置并初始化模型
    
    环境变量:
        LLM_PROVIDER: 指定 LLM 提供商 (google_genai | deepseek)
        LLM_MODEL: 模型名称
        LLM_TEMPERATURE: 温度参数
        GOOGLE_API_KEY: Google API Key
        DEEPSEEK_API_KEY: DeepSeek API Key
        LLM_BASE_URL: DeepSeek API Base URL
    """
    provider = os.getenv("LLM_PROVIDER", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0").strip() or "0")

    # 明确指定 provider 的优先
    if provider == "google_genai":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=google_genai 但未配置 GOOGLE_API_KEY")
        print(f"✅ 使用 Google Gemini 模型: {model or 'gemini-2.5-flash'}")
        return init_chat_model(
            model=model or "gemini-2.5-flash",
            model_provider="google_genai",
            api_key=api_key,
            temperature=temperature,
        )

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=deepseek 但未配置 DEEPSEEK_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        print(f"✅ 使用 DeepSeek 模型: {model or 'deepseek-chat'}")
        return init_chat_model(
            model=model or "deepseek-chat",
            model_provider="deepseek",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    # 未指定 provider 时的回退（兼容旧逻辑）
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        print(f"✅ 使用 Google Gemini 模型: {model or 'gemini-2.5-flash'} (自动选择)")
        return init_chat_model(
            model=model or "gemini-2.5-flash",
            model_provider="google_genai",
            api_key=google_api_key,
            temperature=temperature,
        )

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        print(f"✅ 使用 DeepSeek 模型: {model or 'deepseek-chat'} (自动选择)")
        return init_chat_model(
            model=model or "deepseek-chat",
            model_provider="deepseek",
            api_key=deepseek_api_key,
            base_url=base_url,
            temperature=temperature,
        )

    raise ValueError(
        "未检测到可用的 LLM 配置。请在 .env 文件中设置:\n"
        "  - LLM_PROVIDER (google_genai 或 deepseek)\n"
        "  - 对应的 API Key (GOOGLE_API_KEY 或 DEEPSEEK_API_KEY)"
    )