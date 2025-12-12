import json
import random
import os

# --- 配置参数 ---
NUM_BITS = 50          # 每个二进制数的位数
NUM_SAMPLES = 500000   # 要生成的数据样本总数
OUTPUT_FILENAME = "binary_addition_explainable.jsonl"
# 是否生成双重解释（串行+并行）
DUAL_EXPLANATION = True


def generate_binary_addition_sample(num_bits, dual_explanation=False):
    """
    生成一个二进制加法的样本，包含输入、最终和以及可解释的步骤标签。

    Args:
        num_bits (int): 每个二进制数的位数。
        dual_explanation (bool): 是否生成双重解释（串行+并行）

    Returns:
        dict: 包含 input, output (可解释标签), 和 sum_output (最终和) 的字典。
              如果dual_explanation=True，还包含output_parallel（并行解释）。
    """
    # 1. 生成两个随机的N位二进制数
    num1_int = random.randint(0, 2**num_bits - 1)
    num2_int = random.randint(0, 2**num_bits - 1)

    # 转换为固定长度的二进制字符串
    num1_bin_str = format(num1_int, f'0{num_bits}b')
    num2_bin_str = format(num2_int, f'0{num_bits}b')

    # 拼接成最终的输入字符串
    input_str = num1_bin_str + num2_bin_str

    # 将字符串转为整数列表，方便计算
    bits1 = [int(b) for b in num1_bin_str]
    bits2 = [int(b) for b in num2_bin_str]

    # 2. 逐位计算，生成结果和可解释性标签（串行解释）
    carry = 0
    result_bits_list = []  # 存储每一位的结果
    carry_bits_list = []   # 存储输入到每一位的进位

    # 从最低位（最右边）开始计算
    for i in range(num_bits - 1, -1, -1):
        bit1 = int(num1_bin_str[i])
        bit2 = int(num2_bin_str[i])
        
        # 记录输入到当前位的进位
        carry_bits_list.append(carry)
        
        # 计算当前位的和
        current_sum = bit1 + bit2 + carry
        
        # 计算当前位的结果
        result_bit = current_sum % 2
        result_bits_list.append(result_bit)
        
        # 计算输出到下一位的进位
        carry = current_sum // 2

    # 因为我们是从右到左计算的，所以需要将列表反转
    result_bits_list.reverse()
    carry_bits_list.reverse()

    # 3. 组织成最终的输出格式
    # 串行可解释性标签：[结果位_N-1, ..., 结果位_0] + [进位_N-1, ..., 进位_0]
    # 这是一个长度为 2 * N 的列表
    explainable_output = result_bits_list + carry_bits_list

    # 最终和标签（包含最后一次计算产生的溢出位）
    # 这是一个长度为 N + 1 的列表
    final_sum_output = [carry] + result_bits_list

    result = {
        "input": input_str,
        "output": explainable_output,
        "sum_output": final_sum_output
    }

    # 4. 如果需要双重解释，生成并行解释
    if dual_explanation:
        # 并行解释：XOR（无进位和）+ AND（产生进位的位）
        xor_bits = []
        and_bits = []
        for i in range(num_bits):
            # 计算无进位和 (XOR)
            xor_bits.append(bits1[i] ^ bits2[i])
            # 计算产生进位的潜力 (AND)
            and_bits.append(bits1[i] & bits2[i])
        
        # 并行可解释性标签: [XOR结果位] + [AND结果位]
        # 这代表了并行计算的第一步，后续需要进位传播才能得到最终结果
        parallel_output = xor_bits + and_bits
        result["output_parallel"] = parallel_output

    return result

def main():
    """
    主函数，生成并写入数据集文件。
    """
    print(f"开始生成数据集...")
    print(f"  - 每个二进制数位数: {NUM_BITS}")
    print(f"  - 样本总数: {NUM_SAMPLES}")
    print(f"  - 输出文件: {OUTPUT_FILENAME}")
    print(f"  - 双重解释模式: {DUAL_EXPLANATION}")
    if DUAL_EXPLANATION:
        print(f"    (将同时生成串行解释和并行解释)")

    with open(OUTPUT_FILENAME, 'w') as f:
        for i in range(NUM_SAMPLES):
            # 每10%打印一次进度
            if (i + 1) % (NUM_SAMPLES // 10) == 0:
                print(f"  ...已生成 {i + 1}/{NUM_SAMPLES} ({((i + 1)/NUM_SAMPLES)*100:.0f}%)")
                
            sample = generate_binary_addition_sample(NUM_BITS, dual_explanation=DUAL_EXPLANATION)
            # 将字典转换为JSON字符串并写入文件，后跟换行符
            f.write(json.dumps(sample) + '\n')
            
    print("-" * 20)
    print(f"成功！数据集已保存至 {OUTPUT_FILENAME}")
    print("\n这是一个样本示例:")
    sample = generate_binary_addition_sample(NUM_BITS, dual_explanation=DUAL_EXPLANATION)
    print(json.dumps(sample, indent=2))
    
    if DUAL_EXPLANATION:
        print("\n--- 字段说明 ---")
        print("  input: 两个二进制数拼接")
        print("  output: 串行解释 [结果位] + [进位]")
        print("  output_parallel: 并行解释 [XOR结果] + [AND结果]")
        print("  sum_output: 最终加法结果")

if __name__ == "__main__":
    main()
