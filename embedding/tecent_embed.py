#模型下载
# from modelscope import snapshot_download
# model_dir = snapshot_download('TencentBAC/Conan-embedding-v1')
# 用sentence-transformer就可以调用

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_embedding_model(model_path):
    # 加载本地嵌入模型
    try:
        model = SentenceTransformer(model_path)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败：{e}")
        return

    # 测试文本
    texts = [
        "人工智能是研究如何使计算机模拟人类智能的科学",
        "机器学习是人工智能的一个重要分支，专注于数据驱动的学习算法",
        "气候变化对全球生态系统产生了深远影响",
        "深度学习是机器学习的一种方法，基于人工神经网络"
    ]

    # 生成嵌入向量
    try:
        embeddings = model.encode(texts)
        print(f"嵌入向量生成成功，形状：{embeddings.shape}")  # 应输出 (4, 维度)
    except Exception as e:
        print(f"嵌入向量生成失败：{e}")
        return

    # 计算语义相似度（以第一句为基准）
    print("\n语义相似度（与第一句比较）：")
    base_embedding = embeddings[0].reshape(1, -1)
    for i, text in enumerate(texts):
        sim = cosine_similarity(base_embedding, embeddings[i].reshape(1, -1))[0][0]
        print(f"文本 {i+1}：{text[:30]}...  相似度：{sim:.4f}")

if __name__ == "__main__":
    # 替换为你的模型本地路径
    model_local_path = "/Users/1merci/.cache/modelscope/hub/models/TencentBAC/Conan-embedding-v1"
    test_embedding_model(model_local_path)
