import requests
import json


def get_embedding(text, model="rjmalagon/gte-qwen2-1.5b-instruct-embed-f16", port=11434):
    """
    使用Ollama API获取文本的嵌入向量

    参数:
        text (str): 需要生成嵌入向量的文本
        model (str): 嵌入模型名称
        port (int): Ollama服务端口号，默认11434

    返回:
        list: 文本对应的嵌入向量列表
    """
    # 构建API请求URL
    url = f"http://localhost:{port}/api/embeddings"

    # 准备请求数据
    payload = {
        "model": model,
        "prompt": text
    }

    try:
        # 发送POST请求（必须用POST方法，否则会报405错误）
        response = requests.post(
            url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )

        # 检查请求是否成功
        response.raise_for_status()

        # 解析响应，返回嵌入向量
        result = response.json()
        return result.get("embedding")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP错误: {e}")
        print(f"响应内容: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"连接错误: 无法连接到Ollama服务，请检查服务是否在端口{port}启动")
    except Exception as e:
        print(f"发生错误: {e}")

    return None


if __name__ == "__main__":
    # 示例文本
    sample_texts = [
        "人工智能是计算机科学的一个分支",
        "自然语言处理是人工智能的重要应用领域",
        "嵌入模型能够将文本转换为数值向量"
    ]

    # 为每个文本生成嵌入向量
    for text in sample_texts:
        embedding = get_embedding(text)
        if embedding:
            print(f"文本: {text}")
            print(f"嵌入向量长度: {len(embedding)}")
            print(f"嵌入向量前5个值: {embedding[:5]}...\n")
