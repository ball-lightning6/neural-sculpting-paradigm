import json
import random
import string
import math
from tqdm import tqdm


def solve_freedom_trail_with_path(ring, key):
    """
    使用动态规划解决“自由之路”问题，并回溯以找到最优路径和每一步的操作。
    注意：根据要求，我们不计算按键的+1步。

    Args:
        ring (str): 圆盘上的字符。
        key (str): 需要拼写的关键字。

    Returns:
        tuple: (total_steps, moves_list)
               total_steps (int): 最少的总旋转步数。
               moves_list (list of tuple): 每一步的移动 (direction, steps)，
                                           direction=0为顺时针, 1为逆时针。
    """
    ring_len = len(ring)
    key_len = len(key)

    # 构建字符位置映射，方便查找
    char_positions = {char: [] for char in string.ascii_lowercase}
    for i, char in enumerate(ring):
        char_positions[char].append(i)

    # dp[i][j]: 拼写完 key[:i+1]，且 ring[j] 在12点位置时的最小步数
    # 我们用字典来稀疏存储，只存有效位置
    dp_costs = {pos: 0 for pos in char_positions[key[0]]}
    # path[i][pos] = prev_pos: 记录达到 (i, pos) 状态的最优前驱位置
    dp_path = {}

    def get_rotation_info(start_pos, end_pos, length):
        """计算从 start_pos 到 end_pos 的最短旋转信息"""
        clockwise_dist = (end_pos - start_pos + length) % length
        counter_clockwise_dist = (start_pos - end_pos + length) % length

        if clockwise_dist <= counter_clockwise_dist:
            return (0, clockwise_dist)  # 0 for clockwise
        else:
            return (1, counter_clockwise_dist)  # 1 for counter-clockwise

    # 初始化第一步
    # 从 ring[0] 开始，移动到 key[0] 的各个位置
    initial_pos = 0
    for pos in dp_costs.keys():
        _, steps = get_rotation_info(initial_pos, pos, ring_len)
        dp_costs[pos] = steps
        dp_path[(0, pos)] = initial_pos

    # DP递推
    for i in range(1, key_len):
        prev_char = key[i - 1]
        curr_char = key[i]

        new_dp_costs = {}

        for curr_pos in char_positions[curr_char]:
            min_cost = float('inf')
            best_prev_pos = -1

            for prev_pos in char_positions[prev_char]:
                _, rot_steps = get_rotation_info(prev_pos, curr_pos, ring_len)
                current_total_cost = dp_costs[prev_pos] + rot_steps

                if current_total_cost < min_cost:
                    min_cost = current_total_cost
                    best_prev_pos = prev_pos

            new_dp_costs[curr_pos] = min_cost
            dp_path[(i, curr_pos)] = best_prev_pos

        dp_costs = new_dp_costs

    # --- 回溯路径 ---
    # 1. 找到最后一步的最优位置和总步数
    final_cost = min(dp_costs.values())
    final_pos = min(dp_costs, key=dp_costs.get)

    # 2. 从后往前追溯路径
    path_indices = [0] * key_len
    path_indices[key_len - 1] = final_pos
    for i in range(key_len - 1, 0, -1):
        path_indices[i - 1] = dp_path[(i, path_indices[i])]

    # 3. 根据路径计算每一步的移动
    moves = []
    current_ring_pos = 0
    for i in range(key_len):
        target_ring_pos = path_indices[i]
        direction, steps = get_rotation_info(current_ring_pos, target_ring_pos, ring_len)
        moves.append((direction, steps))
        current_ring_pos = target_ring_pos

    return final_cost, moves


def generate_one_sample(ring_len, key_len, total_steps_bits, move_bits):
    """生成一个单独的、格式化的训练样本"""

    # 1. 生成随机的 ring 和 key
    # 确保 key 中的所有字符都在 ring 中
    alphabet = 'abcdefg'#string.ascii_lowercase
    ring_chars = random.choices(alphabet, k=ring_len)
    ring = "".join(ring_chars)

    # 从 ring 的唯一字符中采样，构建 key
    unique_ring_chars = list(set(ring_chars))
    key = "".join(random.choices(unique_ring_chars, k=key_len))

    # 2. 计算真值
    total_steps, moves = solve_freedom_trail_with_path(ring, key)

    # 3. 格式化输入
    input_str = f"{ring}|{key}"

    # 4. 格式化输出
    output_bits = []

    # Part 1: 总步数
    #output_bits.append(format(total_steps, f'0{total_steps_bits}b'))

    # Part 2: 每一步的移动
    single_move_step_bits = move_bits - 1  # 1 bit for direction
    for direction, steps in moves:
        dir_bit = str(direction)
        steps_bits = format(steps, f'0{single_move_step_bits}b')
        output_bits.append(dir_bit + steps_bits)

    # 如果key比预设的key_len短（虽然这里是固定的），可以用0填充
    # 但由于我们是固定长度，这里不需要

    output_str = "".join(output_bits)
    return {"input": input_str, "output": [int(b) for b in output_str]}


def create_dataset(num_samples, output_path, ring_len, key_len):
    """创建完整的数据集文件"""

    # 计算输出各部分需要的比特数
    # 最坏情况：每一步都转 ring_len / 2
    max_total_steps = math.ceil(key_len * (ring_len / 2))
    total_steps_bits = math.ceil(math.log2(max_total_steps + 1))

    # 单步旋转最多 ring_len / 2
    max_single_move = math.ceil(ring_len / 2)
    move_bits = 1 + math.ceil(math.log2(max_single_move + 1))  # 1 bit for direction

    output_len = key_len * move_bits#total_steps_bits

    print(f"开始生成 {num_samples} 个样本...")
    print(f"固定尺寸: ring长度={ring_len}, key长度={key_len}")
    print(f"输出格式: total_steps( {total_steps_bits} bits) + {key_len} * moves( {move_bits} bits each)")
    print(f"总输出向量长度: {output_len}")

    all_samples = []
    seen_inputs = set()

    with tqdm(total=num_samples, desc="生成样本") as pbar:
        while len(all_samples) < num_samples:
            sample = generate_one_sample(ring_len, key_len, total_steps_bits, move_bits)
            if sample["input"] not in seen_inputs:
                # 确保输出长度一致
                if len(sample["output"])!=output_len: continue
                seen_inputs.add(sample["input"])
                all_samples.append(sample)
                pbar.update(1)

    print("样本生成完毕，开始打乱顺序...")
    random.shuffle(all_samples)

    print(f"开始写入到文件: {output_path}")
    with open(output_path, 'w') as f:
        for sample in tqdm(all_samples, desc="写入文件"):
            f.write(json.dumps(sample) + '\n')

    print("数据集创建成功！")


if __name__=='__main__':
    # --- 配置参数 ---
    NUM_SAMPLES = 1000000
    OUTPUT_FILE = 'freedom_trail_explainable_decoupled_dataset.jsonl'

    # --- 固定尺寸 ---
    RING_LENGTH = 7
    KEY_LENGTH = 5

    # --- 开始执行 ---
    create_dataset(
        num_samples=NUM_SAMPLES,
        output_path=OUTPUT_FILE,
        ring_len=RING_LENGTH,
        key_len=KEY_LENGTH
    )

    # --- 验证生成的文件 (可选) ---
    print("\n--- 验证文件第一行 ---")
    with open(OUTPUT_FILE, 'r') as f:
        first_line = f.readline()
        data = json.loads(first_line)
        print(f"输入 (input): {data['input']}")
        print(f"输出 (output) 长度: {len(data['output'])}")
        # Re-calculate expected length for verification
        max_total_steps = math.ceil(KEY_LENGTH * (RING_LENGTH / 2))
        total_steps_bits = math.ceil(math.log2(max_total_steps + 1))
        max_single_move = math.ceil(RING_LENGTH / 2)
        move_bits = 1 + math.ceil(math.log2(max_single_move + 1))
        expected_len = KEY_LENGTH * move_bits #total_steps_bits
        print(f"期望输出长度: {expected_len}")
        assert len(data['output'])==expected_len, "输出长度与期望不符！"
    print("验证通过！")
