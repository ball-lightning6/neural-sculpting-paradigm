import json
import random
import math
from collections import deque

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 迷宫大小 N x M (限制在较小范围以保证求解器效率)
MAZE_N = 6
MAZE_M = 6

# 特殊点的数量
MAX_BUTTONS = 4  # 机关M
MAX_STONES = 3  # 石头O

DATASET_SIZE = 500000

TRAIN_FILE = f'xunbao_n{MAZE_N}m{MAZE_M}_train.jsonl'
EVAL_FILE = f'xunbao_n{MAZE_N}m{MAZE_M}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
# 输入：[maze_grid] + [start_pos] + [target_pos]
# 每个点用2-bit表示 (0=空地, 1=墙, 2=机关, 3=石头)
# 为了简化，我们直接用一个扁平化的01字符串
# (S,T,M,O用特殊字符或坐标表示)
# 更简单的方式：直接将整个maze字符串作为输入
# 这里我们用最直接的方式：
# 迷宫(N*M*2 bits) + S坐标(logN+logM bits) + T坐标(logN+logM bits)
# 这太复杂了，我们简化一下输入格式
# 输入：一个 N*M 的字符串，用 'S','T','M','O','#','.' 表示
# 输出：cost的二进制表示

# 计算输出位数
# cost最大值理论上很大，但对于小迷宫，我们可以给一个合理的上限
MAX_COST = MAZE_N * MAZE_M * 2
OUTPUT_BITS = math.ceil(math.log2(MAX_COST + 2))  # +1 for 0, +1 for -1

print("=" * 70)
print(f"     寻宝问题 - 数据集生成器 (LeetCode解法最终版)")
print("=" * 70)
print(f"迷宫大小: {MAZE_N}x{MAZE_M}")
print(f"输入格式: {MAZE_N * MAZE_M}个字符的字符串")
print(f"输出格式: {OUTPUT_BITS}个多标签二分类 (-1用0表示)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：求解器 (状态压缩DP) ---
# ==============================================================================

def solve_xunbao(maze):
    n = len(maze)
    m = len(maze[0])

    def in_bound(x, y):
        return 0 <= x < n and 0 <= y < m

    def bfs(x, y):
        dist = [[-1] * m for _ in range(n)]
        dist[x][y] = 0
        q = deque([(x, y)])
        dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
        while q:
            r, c = q.popleft()
            for i in range(4):
                nr, nc = r + dx[i], c + dy[i]
                if in_bound(nr, nc) and maze[nr][nc]!='#' and dist[nr][nc]==-1:
                    dist[nr][nc] = dist[r][c] + 1
                    q.append((nr, nc))
        return dist

    buttons, stones = [], []
    sx, sy, tx, ty = -1, -1, -1, -1
    for i in range(n):
        for j in range(m):
            if maze[i][j]=='M':
                buttons.append((i, j))
            elif maze[i][j]=='O':
                stones.append((i, j))
            elif maze[i][j]=='S':
                sx, sy = i, j
            elif maze[i][j]=='T':
                tx, ty = i, j

    nb = len(buttons)
    if nb==0:
        return bfs(sx, sy)[tx][ty]

    start_dist = bfs(sx, sy)

    # dist[i][j] = button_i 到 button_j 的最短距离
    # dist[i][nb] = start 到 button_i 的最短距离
    # dist[i][nb+1] = button_i 到 target 的最短距离
    dist = [[-1] * (nb + 2) for _ in range(nb)]

    # dd[i] = 从 button_i 出发的bfs结果
    dd = [bfs(bx, by) for bx, by in buttons]

    for i in range(nb):
        dist[i][nb + 1] = dd[i][tx][ty]

        # 计算 start -> O -> button_i
        tmp = float('inf')
        for ox, oy in stones:
            if start_dist[ox][oy]!=-1 and dd[i][ox][oy]!=-1:
                tmp = min(tmp, start_dist[ox][oy] + dd[i][ox][oy])
        dist[i][nb] = tmp if tmp!=float('inf') else -1

        # 计算 button_i -> O -> button_j
        for j in range(i + 1, nb):
            tmp = float('inf')
            for ox, oy in stones:
                if dd[i][ox][oy]!=-1 and dd[j][ox][oy]!=-1:
                    tmp = min(tmp, dd[i][ox][oy] + dd[j][ox][oy])
            dist[i][j] = dist[j][i] = tmp if tmp!=float('inf') else -1

    # 检查是否有无法到达的机关
    for i in range(nb):
        if dist[i][nb]==-1: return -1

    # 状态压缩DP
    dp = [[-1] * nb for _ in range(1 << nb)]
    for i in range(nb):
        dp[1 << i][i] = dist[i][nb]

    for mask in range(1, 1 << nb):
        for i in range(nb):
            if (mask >> i) & 1:
                if dp[mask][i]==-1: continue
                for j in range(nb):
                    if not ((mask >> j) & 1):
                        if dist[i][j]==-1: continue
                        next_mask = mask | (1 << j)
                        if dp[next_mask][j]==-1 or dp[next_mask][j] > dp[mask][i] + dist[i][j]:
                            dp[next_mask][j] = dp[mask][i] + dist[i][j]

    final_mask = (1 << nb) - 1
    ans = float('inf')
    for i in range(nb):
        if dp[final_mask][i]!=-1 and dist[i][nb + 1]!=-1:
            ans = min(ans, dp[final_mask][i] + dist[i][nb + 1])

    return ans if ans!=float('inf') else -1

all_set= set()
# ==============================================================================
# --- 4. 数据集生成主函数 ---
# ==============================================================================
def generate_datasets(num_samples, n, m):
    print("\n--- 开始生成数据集 ---")

    records = []
    for i in range(num_samples):
        while True:
            # 随机生成迷宫
            maze = [['.' for _ in range(m)] for _ in range(n)]

            # 放置墙壁
            for r in range(n):
                for c in range(m):
                    if random.random() < 0.2:
                        maze[r][c] = '#'

            # 放置S, T, M, O
            special_points = random.sample([(r, c) for r in range(n) for c in range(m)], 2 + MAX_BUTTONS + MAX_STONES)

            sr, sc = special_points.pop()
            maze[sr][sc] = 'S'
            tr, tc = special_points.pop()
            maze[tr][tc] = 'T'

            for _ in range(MAX_BUTTONS):
                br, bc = special_points.pop()
                maze[br][bc] = 'M'

            for _ in range(MAX_STONES):
                sr, sc = special_points.pop()
                maze[sr][sc] = 'O'

            maze_str_list = ["".join(row) for row in maze]
            input_str = "".join(maze_str_list)
            if input_str not in all_set:
                all_set.add(input_str)
                break

        min_steps = solve_xunbao(maze_str_list)

        # 编码输入和输出
        input_str = "".join(maze_str_list)

        output_label = min_steps if min_steps!=-1 else 0  # 映射-1到0
        if min_steps!=-1: output_label += 1

        output_binary_str = format(output_label, f'0{OUTPUT_BITS}b')
        output_multilabel = [int(bit) for bit in output_binary_str]

        records.append({"input": input_str, "output": output_multilabel})

        if (i + 1) % 1000==0:
            print(f"已生成 {i + 1} / {num_samples} 条数据...")

    print(f"生成完毕。共 {len(records)} 条数据。")

    # ... (省略写入文件的逻辑)
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
    generate_datasets(DATASET_SIZE, MAZE_N, MAZE_M)