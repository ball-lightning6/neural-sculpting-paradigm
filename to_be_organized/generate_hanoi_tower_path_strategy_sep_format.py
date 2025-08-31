import json
import random
import math
import sys

# ==============================================================================
# --- 1. 核心参数配置 (在这里修改来控制实验) ---
# ==============================================================================

# --- 阶段一：快速验证 ---
# HANOI_N = 4  # 解决4个盘子的汉诺塔问题
# DATASET_SIZE = 20000  # 生成2万条数据用于快速验证
# TRAIN_FILE = f'hanoi_n{HANOO_I_N}_train.jsonl'
# EVAL_FILE = f'hanoi_n{HANOI_N}_eval.jsonl'

# --- 阶段二：正式攻击 ---
HANOI_N = 3  # 解决8个盘子的汉诺塔问题
DATASET_SIZE = 300000  # 生成20万条数据
TRAIN_FILE = f'hanoi_n{HANOI_N}_train.jsonl'
EVAL_FILE = f'hanoi_n{HANOI_N}_eval.jsonl'

# ==============================================================================
# --- 2. 编码与表示定义 (根据N自动计算) ---
# ==============================================================================

# 动作空间：6种可能的移动 (from_peg, to_peg)
# 我们给它们一个固定的索引作为分类标签
ACTIONS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
ACTION_MAP = {action: i for i, action in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)  # 输出是 6 分类

# Token 定义
# 我们需要表示 N 个盘子, 1 个 SEP, 1 个 PAD
# N=8时, 需要表示10个东西, log2(10)≈3.32, 所以需要4位
BITS_PER_TOKEN = math.ceil(math.log2(HANOI_N + 2))

# 特殊 Token 的 ID
SEP_TOKEN_ID = HANOI_N + 1
PAD_TOKEN_ID = 0  # 用0作为PAD比较方便

# 输入序列的固定 Token 长度
INPUT_TOKEN_LEN = HANOI_N + 2  # N个盘子 + 2个分隔符

# 最终输入的总比特长度
TOTAL_INPUT_BITS = INPUT_TOKEN_LEN * BITS_PER_TOKEN

print(f"--- 汉诺塔 (N={HANOI_N}) 数据集生成脚本 ---")
print(f"每个Token使用 {BITS_PER_TOKEN} bits")
print(f"输入序列固定为 {INPUT_TOKEN_LEN} 个 Tokens")
print(f"总输入长度: {TOTAL_INPUT_BITS} bits")
print(f"输出为 {NUM_ACTIONS} 分类问题")
print("-" * 40)


# ==============================================================================
# --- 3. 核心逻辑函数 ---
# ==============================================================================

def hanoi_solver_recursive(n, source, target, auxiliary, moves_list):
    """标准的汉诺塔递归求解器，生成移动步骤列表"""
    if n > 0:
        # 将 n-1 个盘子从 source 移动到 auxiliary
        hanoi_solver_recursive(n - 1, source, auxiliary, target, moves_list)

        # 移动第 n 个盘子从 source 到 target
        # 注意：盘子ID是从1到N，而不是从0开始
        moves_list.append({'disk': n, 'from': source, 'to': target})

        # 将 n-1 个盘子从 auxiliary 移动到 target
        hanoi_solver_recursive(n - 1, auxiliary, target, source, moves_list)


def get_hanoi_solution(n):
    """获取N盘汉诺塔的完整解法"""
    moves = []
    hanoi_solver_recursive(n, 0, 2, 1, moves)
    return moves


def state_to_tokens(pegs):
    """将盘面状态 [[3,2,1], [], []] 转换为 token 列表 [3,2,1, SEP, SEP]"""
    tokens = []
    for i in range(3):  # 遍历三个柱子
        # 盘子ID是从1开始的
        peg_tokens = pegs[i] if pegs[i] else []
        tokens.extend(peg_tokens)
        if i < 2:  # 在前两个柱子后添加分隔符
            tokens.append(SEP_TOKEN_ID)
    return tokens


def tokens_to_binary(tokens):
    """将token列表转换为填充后的二进制字符串"""
    binary_string = ""
    # 先进行填充
    padded_tokens = tokens + [PAD_TOKEN_ID] * (INPUT_TOKEN_LEN - len(tokens))

    for token_id in padded_tokens:
        # 将每个token_id转换为固定位数的二进制
        binary_string += format(token_id, f'0{BITS_PER_TOKEN}b')

    return binary_string


# ==============================================================================
# --- 4. 数据集生成主函数 ---
# ==============================================================================

def generate_hanoi_dataset(n_samples, hanoi_n, output_path):
    """主函数，生成并保存数据集"""

    print(f"正在为 N={hanoi_n} 生成 {n_samples} 条数据...")

    # 1. 首先生成完整的解法序列
    solution_moves = get_hanoi_solution(hanoi_n)

    # 2. 从解法序列中创建 (状态, 下一步动作) 的数据对
    data_pairs = []
    pegs = [list(range(hanoi_n, 0, -1)), [], []]  # 初始状态

    for move in solution_moves:
        # 当前状态
        current_pegs_state = [p[:] for p in pegs]  # 深拷贝当前状态

        # 正确的下一步动作
        action = (move['from'], move['to'])
        action_label = ACTION_MAP[action]

        # 保存数据对
        data_pairs.append({'state': current_pegs_state, 'action': action_label})

        # 更新状态到下一步
        disk_to_move = pegs[move['from']].pop()
        pegs[move['to']].append(disk_to_move)
    print(len(data_pairs))
    # 3. 生成并写入文件
    # 为了数据多样性，我们从所有可能的步骤中随机抽样来构成数据集
    with open(output_path, 'w') as f:
        for i in range(n_samples):
            # 随机选择一个状态-动作对
            pair = random.choice(data_pairs)

            state = pair['state']
            action_label = pair['action']

            # 状态编码
            tokens = state_to_tokens(state)
            input_binary = tokens_to_binary(tokens)

            # 写入JSONL格式
            record = {"input": input_binary, "output": action_label}
            f.write(json.dumps(record) + '\n')

            if (i + 1) % 10000==0:
                sys.stdout.write(f"\r已生成: {i + 1}/{n_samples}")
                sys.stdout.flush()

    print(f"\n数据集 '{output_path}' 生成完成！")


# ==============================================================================
# --- 5. 执行生成 ---
# ==============================================================================

if __name__=="__main__":
    # 生成训练集
    generate_hanoi_dataset(DATASET_SIZE, HANOI_N, TRAIN_FILE)

    # 生成评估集 (训练集的10%)
    #generate_hanoi_dataset(max(1000, DATASET_SIZE // 10), HANOI_N, EVAL_FILE)