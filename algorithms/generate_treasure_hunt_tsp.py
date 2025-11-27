#!/usr/bin/env python3
"""
寻宝游戏（TSP变种）数据集生成器
结合图遍历和状态压缩动态规划的复杂搜索问题
"""

import json
import os
import random
import math
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import cchess  # 假设使用cchess库进行象棋相关操作

# ==============================================================================
# 配置参数
# ==============================================================================
NUM_SAMPLES = 1000  # 默认生成样本数
NUM_PROCESSES = cpu_count()
MAZE_N = 6          # 迷宫行数
MAZE_M = 6          # 迷宫列数
MAX_BUTTONS = 4   # 最大机关数
MAX_STONES = 3    # 最大石头数
OUTPUT_BITS = math.ceil(math.log2(MAZE_N * MAZE_M * 2 + 2))  # 输出位数
OUTPUT_DIR = "treasure_hunt_tsp_dataset"

# ==============================================================================
# 核心算法：寻宝问题求解器（状态压缩DP）
# ==============================================================================
def solve_treasure_hunt(maze):
    """
    寻宝问题求解器 - 结合BFS和状态压缩DP
    解决：从起点S出发，触发所有机关M，利用石头O，到达终点T的最短路径
    """
    n = len(maze)
    m = len(maze[0])
    
    def in_bound(x, y):
        return 0 <= x < n and 0 <= y < m
    
    def bfs(x, y):
        """BFS计算从(x,y)到所有点的最短距离"""
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
    
    # 解析迷宫中的特殊点
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
    
    # 计算所有关键点之间的最短距离
    # dist[i][j] = button_i 到 button_j 的最短距离
    # dist[i][nb] = start 到 button_i 的最短距离
    # dist[i][nb+1] = button_i 到 target 的最短距离
    dist = [[-1] * (nb + 2) for _ in range(nb)]
    
    # 计算起点到各按钮的距离
    for i in range(nb):
        dist[i][nb] = start_dist[buttons[i][0]][buttons[i][1]]
    
    # 计算按钮之间的相互距离
    for i in range(nb):
        button_dist = bfs(buttons[i][0], buttons[i][1])
        for j in range(nb):
            if i != j:
                dist[i][j] = button_dist[buttons[j][0]][buttons[j][1]]
    
    # 计算各按钮到终点的距离
    for i in range(nb):
        dist[i][nb + 1] = bfs(buttons[i][0], buttons[i][1])[tx][ty]
    
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

# ==============================================================================
# 单样本生成任务（用于多进程）
# ==============================================================================
def generate_sample_task(args):
    """多进程任务函数"""
    idx, maze_n, maze_m, max_attempts = args
    try:
        # 生成随机迷宫
        for _ in range(max_attempts):
            # 生成随机迷宫布局
            maze = []
            for i in range(maze_n):
                row = []
                for j in range(maze_m):
                    # 随机生成迷宫元素
                    rand = random.random()
                    if rand < 0.1:
                        row.append('#')  # 墙
                    elif rand < 0.15:
                        row.append('.')  # 空地
                    elif rand < 0.2:
                        row.append('M')  # 机关
                    elif rand < 0.25:
                        row.append('O')  # 石头
                    else:
                        row.append('.')  # 空地
                maze.append(row)
            
            # 添加必要的特殊点
            maze[0][0] = 'S'  # 起点
            maze[maze_n-1][maze_m-1] = 'T'  # 终点
            
            # 添加一些机关和石头
            num_buttons = min(MAX_BUTTONS, maze_n * maze_m // 10)
            num_stones = min(MAX_STONES, maze_n * maze_m // 15)
            
            # 随机放置机关
            button_positions = []
            for _ in range(num_buttons):
                while True:
                    x, y = random.randint(0, maze_n-1), random.randint(0, maze_m-1)
                    if maze[x][y] == '.' and (x, y) not in button_positions:
                        maze[x][y] = 'M'
                        button_positions.append((x, y))
                        break
            
            # 随机放置石头
            stone_positions = []
            for _ in range(num_stones):
                while True:
                    x, y = random.randint(0, maze_n-1), random.randint(0, maze_m-1)
                    if maze[x][y] == '.' and (x, y) not in stone_positions and (x, y) not in button_positions:
                        maze[x][y] = 'O'
                        stone_positions.append((x, y))
                        break
            
            # 检查是否生成了有效的寻宝问题
            result = solve_treasure_hunt(maze)
            if result != -1:  # 有解的情况
                # 构建标准输出格式
                maze_str = ''.join([''.join(row) for row in maze])
                
                sample = {
                    "input": maze_str,
                    "output": {
                        "cost": result,
                        "maze_n": maze_n,
                        "maze_m": maze_m,
                        "num_buttons": len(button_positions),
                        "num_stones": len(stone_positions)
                    },
                    "metadata": {
                        "generation_method": "treasure_hunt_tsp",
                        "algorithm_type": "state_compression_dp",
                        "maze_size": f"{maze_n}x{maze_m}",
                        "special_elements": {
                            "buttons": button_positions,
                            "stones": stone_positions,
                            "start": (0, 0),
                            "target": (maze_n-1, maze_m-1)
                    }
                }
                
                return sample
        
        return None
    except Exception as e:
        print(f"生成样本 {idx} 失败: {e}")
        return None

# ==============================================================================
# 主生成函数
# ==============================================================================
def generate_dataset(num_samples=NUM_SAMPLES, maze_n=MAZE_N, maze_m=MAZE_M, output_dir=OUTPUT_DIR):
    """
    生成完整的数据集
    """
    print("=" * 60)
    print("寻宝游戏（TSP变种）数据集生成器")
    print(f"参数: num_samples={num_samples}, maze_size={maze_n}x{maze_m}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备多进程任务
    tasks = [(i, maze_n, maze_m, 100) for i in range(num_samples)]
    
    print(f"开始生成 {num_samples} 个寻宝迷宫，使用 {NUM_PROCESSES} 个进程...")
    print("正在生成结合图遍历和状态压缩DP的复杂搜索问题...")
    
    # 多进程生成
    dataset = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for result in tqdm(pool.imap_unordered(generate_sample_task, tasks), 
                          total=num_samples, desc="生成进度"):
            if result is not None:
                dataset.append(result)
    
    print(f"成功生成 {len(dataset)} 个寻宝迷宫")
    
    # 保存训练集 (90%)
    train_size = int(len(dataset) * 0.9)
    train_data = dataset[:train_size]
    
    # 保存评估集 (10%)
    eval_data = dataset[train_size:]
    
    # 写入文件
    train_file = os.path.join(output_dir, f"treasure_hunt_tsp_train.jsonl")
    eval_file = os.path.join(output_dir, f"treasure_hunt_tsp_eval.jsonl")
    
    print(f"写入训练集: {len(train_data)} 条记录 -> {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        for record in train_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"写入评估集: {len(eval_data)} 条记录 -> {eval_file}")
    with open(eval_file, 'w', encoding='utf-8') as f:
        for record in eval_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # 生成元数据
    metadata = {
        "dataset_name": "寻宝游戏（TSP变种）数据集",
        "total_samples": len(dataset),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "maze_size": f"{maze_n}x{maze_m}",
        "generation_method": "treasure_hunt_tsp",
        "algorithm_type": "state_compression_dp",
        "output_format": "jsonl",
        "train_file": train_file,
        "eval_file": eval_file
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("=" * 60)
    print("✅ 寻宝迷宫数据集生成完成!")
    print(f"总计: {len(dataset)} 个寻宝迷宫")
    print(f"训练集: {len(train_data)} 个样本")
    print(f"评估集: {len(eval_data)} 个样本")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

# ==============================================================================
# 主执行部分
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="寻宝游戏（TSP变种）数据集生成器")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, 
                       help=f"生成样本数量 (默认: {NUM_SAMPLES})")
    parser.add_argument("--maze_n", type=int, default=MAZE_N,
                       help=f"迷宫行数 (默认: {MAZE_N})")
    parser.add_argument("--maze_m", type=int, default=MAZE_M,
                       help=f"迷宫列数 (默认: {MAZE_M})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                       help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--num_processes", type=int, default=NUM_PROCESSES,
                       help=f"进程数 (默认: {NUM_PROCESSES})")
    
    args = parser.parse_args()
    
    # 设置进程数
    if args.num_processes > 0:
        NUM_PROCESSES = args.num_processes
    
    # 生成数据集
    generate_dataset(
        num_samples=args.num_samples,
        maze_n=args.maze_n,
        maze_m=args.maze_m,
        output_dir=args.output_dir
    )