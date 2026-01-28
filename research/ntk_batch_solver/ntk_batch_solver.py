"""
NTK解析求解器 - 批处理版本
支持一次性测试所有输出位，并计算exact match精度
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import json
import time
import math
import random
import numpy as np


# ==========================================
# 工具函数
# ==========================================
def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_dataset(path, required_samples):
    """验证数据集是否包含足够的样本"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集文件不存在: {path}")

    with open(path, 'r') as f:
        total_lines = sum(1 for _ in f)

    if total_lines < required_samples:
        raise ValueError(
            f"数据集 {path} 只有 {total_lines} 行，"
            f"但需要 {required_samples} 行 (N_TRAIN + N_TEST)"
        )
    return total_lines


# ==========================================
# GPU 向量化相似度公式 (两层无限宽ReLU NTK)
# ==========================================
def compute_ntk_analytic_gpu(X1, X2):
    """
    使用 PyTorch 向量化计算解析NTK相似度
    基于两层无限宽ReLU网络的理论公式

    Args:
        X1: [N, D] 张量
        X2: [M, D] 张量

    Returns:
        K: [N, M] NTK核矩阵
    """
    PI = math.pi

    # 归一化并计算夹角
    X1_norm = torch.norm(X1, dim=1, keepdim=True)
    X2_norm = torch.norm(X2, dim=1, keepdim=True)

    # 余弦相似度
    cos_theta = (X1 @ X2.t()) / (X1_norm @ X2_norm.t())
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    # NTK公式第一项 (arcsin项)
    k_sigma = (1.0 / (2 * PI)) * (X1_norm @ X2_norm.t()) * (torch.sin(theta) + (PI - theta) * cos_theta)

    # NTK公式第二项 (dot product项)
    k_dot = (1.0 / (2 * PI)) * (X1 @ X2.t()) * (PI - theta)

    return k_sigma + k_dot


# ==========================================
# 批处理核心函数
# ==========================================
def solve_ntk_batch(K_train, K_test, Y_train, Y_test, lambda_reg=1e-5):
    """
    批处理求解NTK方程

    Args:
        K_train: [N, N] 训练核矩阵
        K_test: [M, N] 测试核矩阵
        Y_train: [N, B] 训练标签 (B个输出位)
        Y_test: [M, B] 测试标签
        lambda_reg: 正则化系数

    Returns:
        results: 字典，包含每个位的精度和总exact match
    """
    N = K_train.shape[0]
    device = K_train.device

    # 添加正则化
    K_reg = K_train + lambda_reg * torch.eye(N, device=device)

    # 批处理求解所有位: alpha = (K + λI)^(-1) @ Y_train
    # 一次性求解B个线性系统（利用PyTorch的批处理能力）
    alpha = torch.linalg.solve(K_reg, Y_train)  # [N, B]

    # 批处理预测
    Y_pred = K_test @ alpha  # [M, B]
    Y_pred_binary = (Y_pred > 0.5).float()  # 二值化

    # 计算每个位的精确度
    bit_accuracies = {}
    for bit_idx in range(Y_train.shape[1]):
        correct = (Y_pred_binary[:, bit_idx] == Y_test[:, bit_idx]).float().mean()
        bit_accuracies[bit_idx] = correct.item()

    # 计算总的exact match（所有位都正确）
    # exact_match = all bits correct per sample
    exact_matches = torch.all(Y_pred_binary == Y_test, dim=1).float().mean()

    return {
        'bit_accuracies': bit_accuracies,
        'exact_match': exact_matches.item(),
        'avg_bit_accuracy': np.mean(list(bit_accuracies.values()))
    }


# ==========================================
# 主函数
# ==========================================
def main():
    # ========== 配置 ==========
    DATA_PATH = "ca_rule110_layer3_30.jsonl"
    N_TRAIN = 30000
    N_TEST = 1000
    LAMBDA = 1e-5
    DEVICE = "cuda"
    SEED = 42

    # 要测试的输出位
    # None表示测试所有位，或指定列表如 [0, 10, 20, 29]
    BITS_TO_TEST = None

    # ========== 初始化 ==========
    set_seed(SEED)

    print(f"=" * 70)
    print(f"NTK批处理解析求解器")
    print(f"=" * 70)
    print(f"数据集: {DATA_PATH}")
    print(f"训练样本: {N_TRAIN}, 测试样本: {N_TEST}")
    print(f"测试模式: {'所有位' if BITS_TO_TEST is None else f'指定位 {BITS_TO_TEST}'}")
    print(f"=" * 70)

    # ========== 验证数据集 ==========
    total_samples = validate_dataset(DATA_PATH, N_TRAIN + N_TEST)

    # ========== 加载数据 ==========
    print(f"\n[1/4] 加载数据...")
    inputs_all, outputs_all = [], []

    with open(DATA_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i >= N_TRAIN + N_TEST:
                break
            data = json.loads(line)
            inputs_all.append([int(c) for c in data['input']])
            # 注意：output可能是字符串或列表
            out_str = data['output']
            if isinstance(out_str, list):
                outputs_all.append([int(bit) for bit in out_str])
            else:
                outputs_all.append([int(c) for c in out_str])

    # 转换为张量
    X = torch.tensor(inputs_all, dtype=torch.float32).to(DEVICE)  # [N, D]
    Y = torch.tensor(outputs_all, dtype=torch.float32).to(DEVICE)  # [N, B]

    # 分割训练/测试
    x_train, x_test = X[:N_TRAIN], X[N_TRAIN:]
    y_train, y_test = Y[:N_TRAIN], Y[N_TRAIN:]

    # 确定要测试的位
    n_output_bits = y_train.shape[1]
    if BITS_TO_TEST is None:
        # 测试所有位
        bits_to_test = list(range(n_output_bits))
    else:
        bits_to_test = BITS_TO_TEST
        # 验证位索引有效
        if max(bits_to_test) >= n_output_bits:
            raise ValueError(f"指定的位索引 {max(bits_to_test)} 超出输出维度 {n_output_bits}")

    print(f"输出维度: {n_output_bits} 位")
    print(f"实际测试: {len(bits_to_test)} 位")

    # ========== 计算NTK矩阵 ==========
    print(f"\n[2/4] 计算训练NTK矩阵...")
    start_time = time.time()

    # 批量计算以提高GPU利用率
    K_train = compute_ntk_analytic_gpu(x_train, x_train)

    print(f"完成, 耗时: {time.time() - start_time:.2f}s")
    print(f"K_train形状: {K_train.shape}")

    # ========== 计算测试NTK矩阵 ==========
    print(f"\n[3/4] 计算测试NTK矩阵...")
    start_time = time.time()

    K_test = compute_ntk_analytic_gpu(x_test, x_train)

    print(f"完成, 耗时: {time.time() - start_time:.2f}s")
    print(f"K_test形状: {K_test.shape}")

    # ========== 批处理求解 ==========
    print(f"\n[4/4] 批处理求解NTK方程...")
    print(f"求解 {len(bits_to_test)} 个输出位...")
    start_time = time.time()

    # 只保留要测试的位
    y_train_test = y_train[:, bits_to_test]  # [N, B_test]
    y_test_test = y_test[:, bits_to_test]    # [M, B_test]

    # 批处理求解
    results = solve_ntk_batch(
        K_train, K_test, y_train_test, y_test_test, LAMBDA
    )

    solve_time = time.time() - start_time
    print(f"求解完成, 耗时: {solve_time:.2f}s")

    # ========== 输出结果 ==========
    print(f"\n" + "=" * 70)
    print(f"NTK解析求解结果")
    print(f"=" * 70)

    # 显示最重要的结果
    print(f"\n📊 整体性能:")
    print(f"   Exact Match Accuracy: {results['exact_match'] * 100:.4f}%")
    print(f"   平均位精度: {results['avg_bit_accuracy'] * 100:.4f}%")

    # 只显示最低和最高的几个位，避免输出太长
    bit_accuracies = results['bit_accuracies']
    print(f"\n📈 位精度分布:")
    print(f"   最低精度: {min(bit_accuracies.values()) * 100:.4f}% (位 {min(bit_accuracies, key=bit_accuracies.get)})")
    print(f"   最高精度: {max(bit_accuracies.values()) * 100:.4f}% (位 {max(bit_accuracies, key=bit_accuracies.get)})")
    print(f"   精度标准差: {np.std(list(bit_accuracies.values())) * 100:.4f}%")

    # 显示所有位（可选）
    print(f"\n🔍 详细位精度:")
    for bit, acc in bit_accuracies.items():
        print(f"   位 {bit:2d}: {acc * 100:6.2f}%")

    print(f"\n" + "=" * 70)
    print(f"验证: 如果Exact Match > 95%, 证明NTK理论精确成立")
    print(f"=" * 70)


if __name__ == "__main__":
    main()
