"""从原范数实验的 Rule30 终点硬预测复算准确率、种子间共识和共同错误。不训练网络。"""
import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SIZES = (600, 750, 900, 1200)
ARMS = ("baseline", "high_norm", "low_norm")
SEEDS = range(4)
PAIRS = tuple(itertools.combinations(SEEDS, 2))
WIDTH, PROBES, STEP = 30, 4096, 20000
MASK = (1 << WIDTH) - 1
FIELDS = ("n", "probe_id", "target", *(f"{arm}_{s}" for arm in ARMS for s in SEEDS))
SHARED_DATA_HASH = "b3c97334dbc10527ddfda21dd7e29eb828a9d417103f2e466b4eda3d2566a30b"
HASHES = {
    "high_norm": ("45c643f19d44a87db449e958c31705f843c12268c9a836397f7eff05d1d92b61",
                  "e2da35ae41b5e4d40bb8f8366ba0d45b0eeb4de32e31655999e00e193a4e5346",
                  "6f0392fe9071c7d59373319129bf9b785b2c27e0f410181013b3437f4bb622ae"),
    "low_norm": ("a48ed4b5508f05a7a2053df84fb8dc848c35e91d914c66ca3d860886575a9e17",
                 "80a75bbfda07e3d5c3f3b66ba6345ccc5cf597360d239c1c1470a341c93dd05e",
                 "99e903329941c2946f2cace390221ed76a14af4e16b3440eb6a5961ecd0616fc"),
}


def rule30(x):
    left = (x >> 1) | ((x & 1) << (WIDTH - 1))
    right = ((x << 1) & MASK) | (x >> (WIDTH - 1))
    return left ^ (x | right)


def from_original(high, low):
    import numpy as np

    predictions = {}
    ids = None
    powers = 1 << np.arange(WIDTH-1, -1, -1, dtype=np.uint32)
    for folder, treatment in zip((high, low), ARMS[1:]):
        for name, digest in zip(("DATA.npz", "LAYOUT.json", "PREDICTIONS.npz", "HISTORY.json"),
                                (SHARED_DATA_HASH, *HASHES[treatment])):
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"历史来源不一致：{folder/name}")
        with np.load(folder/"DATA.npz", allow_pickle=False) as data:
            saved_ids = tuple(map(int, data["probe_ids"]))
            if set(saved_ids).intersection(data["train_ids"].tolist() + data["monitor_ids"].tolist()):
                raise ValueError("probe 与其他划分重叠")
        if ids is not None and ids != saved_ids:
            raise ValueError("两个运行的探针顺序不同")
        ids = saved_ids
        target = np.array([rule30(x) for x in ids], dtype=np.uint32)
        with np.load(folder/"PREDICTIONS.npz", allow_pickle=False) as saved:
            if int(saved["steps"][-1]) != STEP:
                raise ValueError("未找到正式终点")
            unpacked = np.unpackbits(saved["probe"][-1], axis=-1, count=WIDTH, bitorder="little")
        layout = json.loads((folder/"LAYOUT.json").read_text("utf-8"))
        history = json.loads((folder/"HISTORY.json").read_text("utf-8"))[-1]
        if unpacked.shape != (len(layout), PROBES, WIDTH) or len(history) != len(layout):
            raise ValueError("预测或历史的维度不符")
        found = set()
        for model in layout:
            key = model["n"], model["loss"], model["seed"]
            i = model["model_index"]
            h = history[i]
            if any(h[k] != model[k] for k in ("model_index", "n", "loss", "seed")) or h["step"] != STEP:
                raise ValueError(f"模型编号不匹配：{key}")
            words = unpacked[i].astype(np.uint32) @ powers
            if not h["fitted"] or h["probe_exact"] != float(np.mean(words == target)):
                raise ValueError(f"终点拟合或原整串准确率不匹配：{key}")
            if key in found or (key in predictions and (key[1] != "baseline" or
                                                        not np.array_equal(words, predictions[key]))):
                raise ValueError(f"重复记录或两次基线预测不同：{key}")
            predictions[key] = words
            found.add(key)
        if found != set(itertools.product(SIZES, ("baseline", treatment), SEEDS)):
            raise ValueError("原实验条件不完整")
    return [(n, x, rule30(x), *(int(predictions[n, a, s][i]) for a in ARMS for s in SEEDS))
            for n in SIZES for i, x in enumerate(ids)]


def csv_text(fields, rows):
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(fields)
    writer.writerows(rows)
    return out.getvalue()


def read_public(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if tuple(next(reader)) != FIELDS:
            raise ValueError("逐输入表的列名不符")
        return [tuple(map(int, row)) for row in reader]


def tally(rows):
    seen = {n: set() for n in SIZES}
    votes = {(n, arm): [0]*5 for n, arm in itertools.product(SIZES, ARMS)}
    # 每一对先分成互斥的五类，再统计逐位一致数。共同答错并不等于预测相同。
    counts = {(n, a, i, j): [0]*6 for n, a in votes for i, j in PAIRS}
    for n, x, y, *outputs in rows:
        if n not in seen or x in seen[n] or not 0 <= x <= MASK or y != rule30(x):
            raise ValueError(f"样本重复、越界或规则标签不符：{n}, {x}")
        if len(outputs) != 12 or not all(0 <= v <= MASK for v in outputs):
            raise ValueError("需要十二个 30 位输出")
        seen[n].add(x)
        for arm_id, arm in enumerate(ARMS):
            words = outputs[4*arm_id:4*arm_id+4]
            votes[n, arm][words.count(y)] += 1
            for i, j in PAIRS:
                u, v = words[i], words[j]
                outcome = (0 if v == y else 1) if u == y else (2 if v == y else 3 if u == v else 4)
                bins = counts[n, arm, i, j]
                bins[outcome] += 1
                bins[5] += WIDTH - (u ^ v).bit_count()
    if any(len(block) != PROBES or block != seen[SIZES[0]] for block in seen.values()):
        raise ValueError("各样本档应使用同一组 4096 个 probe")
    pairs, groups = [], []
    for (n, a), h in votes.items():
        bins = [counts[n, a, i, j] for i, j in PAIRS]
        for (i, j), b in zip(PAIRS, bins):
            if sum(b[:5]) != PROBES:
                raise ValueError("每对种子的类别计数不完整")
            pairs.append((n, a, i, j, *b))
        both, only_i, only_j, same_wrong, diff_wrong, bit_same = map(sum, zip(*bins))
        correct = sum(k*v for k, v in enumerate(h))
        if 3*correct != 2*both+only_i+only_j:
            raise ValueError("逐输入正确票数与模型对统计不符")
        groups.append((n, a, *h, correct, both+same_wrong, both, same_wrong, diff_wrong, bit_same))
    return pairs, groups


PAIR_FIELDS = ("n", "arm", "seed_i", "seed_j", "both_correct", "only_i_correct", "only_j_correct",
               "both_wrong_same", "both_wrong_different", "agreeing_bits")
GROUP_FIELDS = ("n", "arm", *(f"inputs_with_{i}_correct" for i in range(5)), "correct_model_rings",
                "agreeing_pair_rings", "both_correct", "both_wrong_same", "both_wrong_different", "agreeing_pair_bits")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, nargs=2, metavar=("HIGH_DIR", "LOW_DIR"))
    p.add_argument("--export", action="store_true", help="从核验过的原件生成公开数据")
    args = p.parse_args()
    if args.export and not args.sources:
        p.error("--export 需要 --sources")
    original = from_original(*args.sources) if args.sources else None
    data = HERE/"endpoints.csv.gz"
    if args.export:
        tally(original)
        with data.open("wb") as f, gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as gz:
            gz.write(csv_text(FIELDS, original).encode())
    rows = read_public(data)
    if original is not None and original != rows:
        raise ValueError("公开预测与原件不一致")
    pairs, groups = tally(rows)
    for name, fields, records in (("seed_pairs.csv", PAIR_FIELDS, pairs), ("group_counts.csv", GROUP_FIELDS, groups)):
        text = csv_text(fields, records)
        if args.export:
            (HERE/name).write_bytes(text.encode())
        elif (HERE/name).is_file() and (HERE/name).read_text("utf-8") != text:
            raise ValueError(f"公开计数与重算不符：{name}")
    print("n arm | accuracy / agreement / different-if-both-wrong (%) | inputs with 0..4 correct seeds")
    for n, arm, *counts in groups:
        h = counts[:5]
        correct, same, _, wrong_same, wrong_diff, _ = counts[5:]
        wrong = wrong_same+wrong_diff
        conditional = f"{100*wrong_diff/wrong:.2f}" if wrong else "NA"
        print(f"{n} {arm:9} | {100*correct/(4*PROBES):.2f} / {100*same/(6*PROBES):.2f} / {conditional} | {h}")


if __name__ == "__main__":
    main()
