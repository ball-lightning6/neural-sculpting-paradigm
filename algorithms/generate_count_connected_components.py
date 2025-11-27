import json
import random
import math

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 图的节点数 N
GRAPH_SIZE_N = 8  # 从一个中等大小开始
# 控制图的边的密度，值越小，图越稀疏，连通分量可能越多
EDGE_PROBABILITY = 0.1

DATASET_SIZE = 500000

TRAIN_FILE = f'connected_components_n{GRAPH_SIZE_N}_train.jsonl'
EVAL_FILE = f'connected_components_n{GRAPH_SIZE_N}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = GRAPH_SIZE_N * GRAPH_SIZE_N
# 输出是连通分量的数量，最多是N个
OUTPUT_BITS = math.ceil(math.log2(GRAPH_SIZE_N))

print("=" * 70)
print(f"     图连通分量计数 - 数据集生成器")
print("=" * 70)
print(f"图大小: {GRAPH_SIZE_N}x{GRAPH_SIZE_N}")
print(f"输入格式: {INPUT_LEN}个'0'/'1'的邻接矩阵")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (代表连通分量数)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：求解器 (基于BFS/DFS的图遍历) ---
# ==============================================================================
all_set=set()
def generate_graph(n, p):
    """随机生成一个无向图的邻接矩阵"""
    while True:
        graph = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                if i==j:
                    graph[i][j] = 1
                elif random.random() < p:
                    graph[i][j] = 1
                    graph[j][i] = 1
        graph_tuple = tuple(tuple(x) for x in graph)
        if graph_tuple not in all_set:
            all_set.add(graph_tuple)
            return graph


def solve_connected_components(graph):
    """
    使用BFS或DFS计算图的连通分量数。
    """
    n = len(graph)
    if n==0:
        return 0

    visited = [False] * n
    count = 0

    for i in range(n):
        if not visited[i]:
            count += 1
            # 从节点i开始，进行一次图遍历，标记所有能到达的节点
            q = [i]
            visited[i] = True
            while q:
                u = q.pop(0)
                for v in range(n):
                    if graph[u][v]==1 and not visited[v]:
                        visited[v] = True
                        q.append(v)
    return count


def process_sample(n, p, output_bits):
    """
    生成一个完整的 (输入, 输出) 数据对。
    """
    # 1. 生成输入图
    graph = generate_graph(n, p)
    input_str = "".join(str(cell) for row in graph for cell in row)

    # 2. 计算连通分量数
    num_components = solve_connected_components(graph)

    # 3. 编码输出
    output_binary_str = format(num_components-1, f'0{output_bits}b')
    output_multilabel = [int(bit) for bit in output_binary_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, n, p, output_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    for i in range(num_samples):
        records.append(process_sample(n, p, output_bits))
        if (i + 1) % 10000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # --- 写入文件 ---
    # ... (省略)
    random.shuffle(records)
    train_size = int(len(records) * 1)#0.9)
    train_data, eval_data = records[:train_size], records[train_size:]

    def write_to_file(data, path, name):
        print(f"\n正在写入 {len(data)} 条{name}训练数据到 '{path[0]}'...")
        with open(path[0], 'w') as f:
            for record in data: f.write(json.dumps(record) + '\n')
        print(f"正在写入 {len(eval_data)} 条{name}评估数据到 '{path[1]}'...")
        with open(path[1], 'w') as f:
            for record in eval_data: f.write(json.dumps(record) + '\n')

    write_to_file(records, (TRAIN_FILE, EVAL_FILE), "")
    print("\n所有数据集生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================
if __name__=="__main__":
    generate_datasets(DATASET_SIZE, GRAPH_SIZE_N, EDGE_PROBABILITY, OUTPUT_BITS)