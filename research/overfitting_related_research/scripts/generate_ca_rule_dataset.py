import json
import random
from pathlib import Path

from tqdm import tqdm


# =========================
# 数据集配置
# =========================
NUM_SAMPLES = 300000
LENGTH = 30
LAYERS = 1

# 支持任意 elementary cellular automaton rule：0-255。
# 例如 110、30。
RULE_NUMBER = 30

# 数据随机种子：控制随机输入样本。
DATA_SEED = 20260711

# 输入 shuffle 的含义：
# jsonl 里的 input 仍然保存原始 x；
# 但是生成 output 时，规则实际作用在 P(x) 上。
SHUFFLE_INPUT = False

# 输出 shuffle 是可选的。它只是在规则演化完成后，再固定重排输出位。
SHUFFLE_OUTPUT = False

# permutation 随机种子：只控制固定置换，不控制输入样本。
PERMUTATION_SEED = 20260701

# 如果你想手动指定 permutation，就填一个 0 到 LENGTH-1 的列表。
# None 表示按 PERMUTATION_SEED 随机生成。
INPUT_PERMUTATION = None
OUTPUT_PERMUTATION = None

OUTPUT_DIR = "research/overfitting_related_research/datasets"

# Direct execution generates the datasets required by sweep_task_difficulty_plateau.py.
DEFAULT_LAYER_SUITE = (1, 2, 3)


NEIGHBORHOODS = ("111", "110", "101", "100", "011", "010", "001", "000")


def build_rule_table(rule_number):
    if not 0 <= rule_number <= 255:
        raise ValueError("RULE_NUMBER 必须在 0 到 255 之间。")
    bits = format(rule_number, "08b")
    return dict(zip(NEIGHBORHOODS, bits))


def make_permutation(length, seed):
    rng = random.Random(seed)
    perm = list(range(length))
    rng.shuffle(perm)
    return perm


def validate_permutation(perm, length, name):
    if len(perm) != length:
        raise ValueError(f"{name} 长度必须等于 LENGTH={length}。")
    if sorted(perm) != list(range(length)):
        raise ValueError(f"{name} 必须是 0 到 {length - 1} 的一个排列。")


def apply_permutation(seq, perm):
    return "".join(seq[i] for i in perm)


def evolve_once(state, rule_table):
    # 循环边界条件：左右首尾相连。
    padded = state[-1] + state + state[0]
    return "".join(rule_table[padded[i:i + 3]] for i in range(len(state)))


def evolve(state, rule_table, layers):
    current = state
    for _ in range(layers):
        current = evolve_once(current, rule_table)
    return current


def build_output_path():
    parts = [
        f"ca_rule{RULE_NUMBER}",
        f"layer{LAYERS}",
        f"len{LENGTH}",
        f"n{NUM_SAMPLES}",
    ]
    if SHUFFLE_INPUT:
        parts.append(f"inshuffle{PERMUTATION_SEED}")
    if SHUFFLE_OUTPUT:
        parts.append(f"outshuffle{PERMUTATION_SEED}")
    return Path(OUTPUT_DIR) / ("_".join(parts) + ".jsonl")


def generate_dataset():
    output_path = build_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rule_table = build_rule_table(RULE_NUMBER)
    rng = random.Random(DATA_SEED)

    input_perm = None
    output_perm = None
    if SHUFFLE_INPUT:
        input_perm = INPUT_PERMUTATION
        if input_perm is None:
            input_perm = make_permutation(LENGTH, PERMUTATION_SEED)
        validate_permutation(input_perm, LENGTH, "INPUT_PERMUTATION")

    if SHUFFLE_OUTPUT:
        output_perm = OUTPUT_PERMUTATION
        if output_perm is None:
            output_perm = make_permutation(LENGTH, PERMUTATION_SEED + 1)
        validate_permutation(output_perm, LENGTH, "OUTPUT_PERMUTATION")

    with output_path.open("w", encoding="utf-8") as f:
        for _ in tqdm(range(NUM_SAMPLES), desc=f"rule {RULE_NUMBER}, layer {LAYERS}"):
            input_seq = "".join(rng.choice("01") for _ in range(LENGTH))

            working_seq = input_seq
            if input_perm is not None:
                working_seq = apply_permutation(working_seq, input_perm)

            output_seq = evolve(working_seq, rule_table, LAYERS)

            if output_perm is not None:
                output_seq = apply_permutation(output_seq, output_perm)

            sample = {
                "input": input_seq,
                "output": [int(bit) for bit in output_seq],
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    meta_path = output_path.with_suffix(".meta.json")
    metadata = {
        "num_samples": NUM_SAMPLES,
        "length": LENGTH,
        "layers": LAYERS,
        "rule_number": RULE_NUMBER,
        "rule_table": rule_table,
        "data_seed": DATA_SEED,
        "shuffle_input": SHUFFLE_INPUT,
        "shuffle_output": SHUFFLE_OUTPUT,
        "permutation_seed": PERMUTATION_SEED,
        "input_permutation": input_perm,
        "output_permutation": output_perm,
        "output_path": str(output_path),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"数据集已保存：{output_path}")
    print(f"元数据已保存：{meta_path}")


def generate_default_suite():
    global LAYERS
    original_layers = LAYERS
    for layers in DEFAULT_LAYER_SUITE:
        LAYERS = layers
        generate_dataset()
    LAYERS = original_layers


if __name__ == "__main__":
    generate_default_suite()
