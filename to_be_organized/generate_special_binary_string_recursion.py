import json
import random

# ==============================================================================
# --- 1. 核心参数配置 ---
# ==============================================================================
# 输入的二进制字符串的长度 (必须是偶数，且不大)
STRING_LENGTH_N = 30  # 从一个小而有意义的长度开始
DATASET_SIZE = 300000

TRAIN_FILE = f'special_binary_string_n{STRING_LENGTH_N}_train.jsonl'
EVAL_FILE = f'special_binary_string_n{STRING_LENGTH_N}_eval.jsonl'

# ==============================================================================
# --- 2. 编码定义 ---
# ==============================================================================
INPUT_LEN = STRING_LENGTH_N
OUTPUT_LEN = STRING_LENGTH_N

print("=" * 70)
print(f"     特殊二进制序列 - 数据集生成器")
print("=" * 70)
print(f"任务: 对一个{INPUT_LEN}-bit的特殊二进制序列，找到字典序最大的结果")
print(f"输入格式: {INPUT_LEN}个'0'/'1'的字符序列")
print(f"输出格式: {OUTPUT_LEN}个多标签二分类 (代表最终的字符串)")
print("=" * 70)


# ==============================================================================
# --- 3. 核心逻辑：求解器 (基于递归和贪心的解法) ---
# ==============================================================================
def is_special(s):
    """检查一个字符串是否是“特殊的”"""
    if s.count('1')!=s.count('0'):
        return False
    balance = 0
    for char in s:
        if char=='1':
            balance += 1
        else:
            balance -= 1
        if balance < 0:
            return False
    return True


def generate_special_string(n):
    """
    随机生成一个长度为n的特殊二进制序列。
    这是一个经典算法问题，与“有效括号”同构。
    """
    if n % 2!=0: return None

    # 采用随机游走的方式生成
    while True:
        s = []
        ones = n // 2
        zeros = n // 2
        balance = 0
        valid = True
        for _ in range(n):
            # 优先放1，或者当0和1数量一样时
            if (random.random() < 0.5 and ones > 0) or zeros==0:
                if balance < n / 2:  # 避免1过多
                    s.append('1')
                    ones -= 1
                    balance += 1
                else:  # 只能放0
                    if balance > 0:
                        s.append('0')
                        zeros -= 1
                        balance -= 1
                    else:  # 生成失败
                        valid = False;
                        break
            else:  # 放0
                if balance > 0:
                    s.append('0')
                    zeros -= 1
                    balance -= 1
                else:  # 只能放1
                    if ones > 0:
                        s.append('1')
                        ones -= 1
                        balance += 1
                    else:  # 生成失败
                        valid = False;
                        break
            if balance < 0:  # 任何前缀的1都不能少于0
                valid = False;
                break
        if valid and balance==0:  # 最终1和0数量要相等
            return "".join(s)


def solve_special_binary_string(s):
    """
    这是一个困难的递归/贪心问题。
    正确的解法是递归地处理。
    对于 s = '1' + A + '0' + B，其中A和B也是特殊序列，
    最终结果是 '1' + makeLargestSpecial(A) + '0' + makeLargestSpecial(B)
    然后对所有可能的分解，进行排序，取字典序最大的。
    """
    n = len(s)
    if n==0:
        return ""

    # 找到所有可以分解出的特殊子串
    count = 0
    i = 0
    res = []
    for j in range(n):
        if s[j]=='1':
            count += 1
        else:
            count -= 1
        if count==0:
            # s[i+1:j] 是一个内部的特殊子串A
            # 递归地处理它
            res.append('1' + solve_special_binary_string(s[i + 1:j]) + '0')
            i = j + 1

    # 对所有找到的特殊子串，按字典序降序排列，然后拼接
    res.sort(reverse=True)
    return "".join(res)

all_set = set()
def process_sample(n_bits):
    """生成一个完整的 (输入, 输出) 数据对。"""
    # 1. 生成一个特殊的输入字符串
    input_str = None
    while not input_str:
        input_str = generate_special_string(n_bits)
        if input_str is not None and input_str!='' and input_str not in all_set:
            all_set.add(input_str)
        else:
            input_str=''

    # 2. 计算最优解
    output_str = solve_special_binary_string(input_str)

    # 3. 编码输出
    output_multilabel = [int(bit) for bit in output_str]

    return {"input": input_str, "output": output_multilabel}


# ==============================================================================
# --- 4. 主生成函数 ---
# ==============================================================================
def generate_datasets(num_samples, n_bits):
    print("\n--- 开始生成数据集 ---")

    records = []
    # 在这个任务中，随机输入几乎不可能重复
    for i in range(num_samples):
        records.append(process_sample(n_bits))

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
    generate_datasets(DATASET_SIZE, STRING_LENGTH_N)