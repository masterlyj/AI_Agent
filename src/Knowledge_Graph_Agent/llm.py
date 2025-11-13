import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

def get_llm():
    """
    从 .env 环境变量读取 LLM 配置并初始化模型
    
    环境变量:
        LLM_PROVIDER: 指定 LLM 提供商 (google_genai | deepseek | vllm)
        LLM_MODEL: 模型名称
        LLM_TEMPERATURE: 温度参数
        GOOGLE_API_KEY: Google API Key
        DEEPSEEK_API_KEY: DeepSeek API Key
        LLM_BASE_URL: API Base URL (用于 deepseek 和 vllm)
        VLLM_API_KEY: vLLM API Key (可选，默认为"EMPTY")
    
    Returns:
        tuple: (llm_instance, model_name) - LLM实例和模型名称
    """
    provider = os.getenv("LLM_PROVIDER", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0").strip() or "0")

    # 明确指定 provider 的优先
    if provider == "google_genai":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=google_genai 但未配置 GOOGLE_API_KEY")
        model_name = model or "gemini-2.5-flash"
        llm = init_chat_model(
            model=model_name,
            model_provider="google_genai",
            api_key=api_key,
            temperature=temperature,
        )
        return llm, model_name

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=deepseek 但未配置 DEEPSEEK_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        model_name = model or "deepseek-chat"
        llm = init_chat_model(
            model=model_name,
            model_provider="deepseek",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )
        return llm, model_name
    
    if provider == "vllm":
        base_url = os.getenv("LLM_BASE_URL")
        if not base_url:
            raise ValueError("LLM_PROVIDER=vllm 但未配置 LLM_BASE_URL")
        if not model:
            raise ValueError("LLM_PROVIDER=vllm 但未配置 LLM_MODEL")
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        model_name = model
        llm = init_chat_model(
            model=model_name,
            model_provider="openai",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )
        return llm, model_name

    # 未指定 provider 时的回退
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        model_name = model or "gemini-2.5-flash"
        llm = init_chat_model(
            model=model_name,
            model_provider="google_genai",
            api_key=google_api_key,
            temperature=temperature,
        )
        return llm, model_name

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        model_name = model or "deepseek-chat"
        llm = init_chat_model(
            model=model_name,
            model_provider="deepseek",
            api_key=deepseek_api_key,
            base_url=base_url,
            temperature=temperature,
        )
        return llm, model_name

    raise ValueError(
        "未检测到可用的 LLM 配置。请在 .env 文件中设置:\n"
        "  - LLM_PROVIDER (google_genai, deepseek 或 vllm)\n"
        "  - 对应的 API Key 和配置\n"
        "  - vLLM需要: LLM_MODEL, LLM_BASE_URL"
    )