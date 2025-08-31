import random
import json
import networkx as nx
from tqdm import tqdm

def generate_sample():
    size = 5
    # 生成邻接矩阵（无向图，0-1边）
    matrix = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(i+1, size):
            if random.random() < 0.3:  # 控制稀疏度
                matrix[i][j] = matrix[j][i] = 1

    # 转成 NetworkX 图用于查路径
    G = nx.Graph()
    G.add_nodes_from(range(size))
    for i in range(size):
        for j in range(size):
            if matrix[i][j] == 1:
                G.add_edge(i, j)

    start, end = random.randint(0, 4), random.randint(0, 4)
    has_path = int(nx.has_path(G, start, end))

    # 构造 input 字符串
    matrix_bits = ''.join(str(bit) for row in matrix for bit in row)
    input_str = matrix_bits + f";{start}{end}"
    return {"input": input_str, "output": [int(has_path)]}

def generate_dataset(filename, num_samples=10000):
    with open(filename, 'w') as f:
        for _ in tqdm(range(num_samples)):
            sample = generate_sample()
            f.write(json.dumps(sample) + '\n')

# 使用示例
generate_dataset("graph_path_reasoning_val.jsonl", num_samples=300000)
