import json
import numpy as np
import random
from tqdm import tqdm

# === 参数设置 ===
n_cells = 15              # 元胞数
n_states = 4              # 每格状态数（0~3），对应2 bit编码
n_samples = 500000         # 样本数量
steps = 12                 # 演化步数（这里只生成一步输入输出）
output_file = 'cca_ring_train_15.jsonl'  # 输出文件名

# === CCA 演化函数 ===
def cca1d_step(state, n_states=4):
    new_state = state.copy()
    for i in range(len(state)):
        left = state[(i - 1) % len(state)]
        right = state[(i + 1) % len(state)]
        target = (state[i] + 1) % n_states
        if left == target or right == target:
            new_state[i] = target
    return new_state

# === 样本生成函数 ===
def generate_sample(n_cells, n_states, steps):
    state = np.random.randint(0, n_states, size=n_cells)
    input_bits = ''.join(f'{s:02b}' for s in state)  # 每格2bit编码为01字符串

    evolved = state.copy()
    for _ in range(steps):
        evolved = cca1d_step(evolved, n_states)
    output_bits = list(map(int,''.join(f'{s:02b}' for s in evolved)))
    changed = (evolved != state).astype(int).tolist()
    return {"input": input_bits, "output": output_bits}

# === 主程序：生成数据集 ===
with open(output_file, 'w') as f:
    for _ in tqdm(range(n_samples)):
        sample = generate_sample(n_cells, n_states, steps)
        f.write(json.dumps(sample) + '\n')

print(f"✅ 已生成 {n_samples} 条训练数据，保存在：{output_file}")
