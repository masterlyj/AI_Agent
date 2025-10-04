import numpy as np
import faiss
import time

print("=== FAISS基本功能测试 ===")

# 生成测试数据
d = 64  # 向量维度
nb = 10000  # 数据库向量数量
nq = 5  # 查询向量数量

print(f"生成随机数据: {nb}个{d}维向量")
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')  # 数据库向量
xq = np.random.random((nq, d)).astype('float32')  # 查询向量

# 创建基础索引
print("创建IndexFlatL2索引...")
index = faiss.IndexFlatL2(d)  # 使用L2距离（欧氏距离）
print(f"索引是否已训练: {index.is_trained}")

# 添加数据到索引
print("添加数据到索引...")
index.add(xb)
print(f"索引中的向量数量: {index.ntotal}")

# 执行搜索
k = 4  # 查找最近的4个邻居
print(f"执行搜索，查找每个查询向量的前{k}个最近邻...")
start_time = time.time()
D, I = index.search(xq, k)
search_time = time.time() - start_time

print(f"搜索耗时: {search_time:.3f}秒")
print("\n搜索结果:")
print("距离矩阵D (越小越相似):")
print(D)
print("\n索引矩阵I (对应数据库中的位置):")
print(I)

# 验证结果正确性
print("\n=== 结果验证 ===")
for i in range(nq):
    print(f"查询向量 {i} 的最近邻索引: {I[i]}")
    print(f"对应距离: {D[i]}")
    
    # 手动计算第一个结果的距离进行验证
    if i == 0:
        manual_dist = np.linalg.norm(xq[0] - xb[I[0][0]])
        print(f"手动验证距离: {manual_dist:.6f}")
        print(f"FAISS计算距离: {D[0][0]:.6f}")
        print(f"距离计算是否正确: {np.isclose(manual_dist, D[0][0])}")

def test_ivf_index():
    print("\n=== IVF高级索引测试 ===")
    
    # 使用相同的测试数据
    d = 64
    nlist = 100  # 聚类中心数量
    
    # 创建IVF索引
    quantizer = faiss.IndexFlatL2(d)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    
    print("训练IVF索引...")
    index_ivf.train(xb)
    index_ivf.add(xb)
    index_ivf.nprobe = 10  # 搜索的聚类中心数量
    
    print("执行IVF搜索...")
    start_time = time.time()
    D_ivf, I_ivf = index_ivf.search(xq, 4)
    ivf_time = time.time() - start_time
    
    print(f"IVF搜索耗时: {ivf_time:.3f}秒")
    print("IVF搜索结果索引:", I_ivf)
    
    return index_ivf
test_ivf_index()