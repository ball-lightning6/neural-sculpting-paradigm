"""从 Rule30 逐次记录复算已测样本门槛；不运行训练。"""
import argparse
import csv
from hashlib import sha256
import io
import itertools
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
ARCHIVE_HASH = "53ebd275548d653003c2944c2601609900bdd2e02d833ec45d13e9fa466cdbe4"
WIDTHS = (128, 256, 512, 1024, 2048)
SIZES = (256, 384, 512, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1536, 2048)
LEVELS = (.001, .0003, .0001)
FIELDS = ("width", "train_count", "model_seed", "target_raw_bce", "train_exact_before",
          "train_exact_after", "matched_probe_exact_accuracy")


def from_archive(path):
    if sha256(path.read_bytes()).hexdigest() != ARCHIVE_HASH:
        raise ValueError("不是 2026-08-20 正式宽度扫描结果包")
    with ZipFile(path) as z:
        rows = [json.loads(line) for line in z.read("matched_loss_crossings.jsonl").splitlines()]
        return [{k: r[k] for k in FIELDS} for r in rows if r["target_raw_bce"] in LEVELS]


def summary(rows):
    scores = {}
    for r in rows:
        key = (int(r["width"]), int(r["train_count"]), int(r["model_seed"]), float(r["target_raw_bce"]))
        score = float(r["matched_probe_exact_accuracy"])
        if key in scores or not 0 <= score <= 1 or any(float(r[k]) != 1 for k in FIELDS[4:6]):
            raise ValueError(f"重复条件、非法准确率或穿越前后未完全拟合：{key}")
        scores[key] = score
    if set(scores) != set(itertools.product(WIDTHS, SIZES, range(3), LEVELS)):
        raise ValueError("未覆盖 5 个宽度 × 14 档输入数 × 3 个种子 × 3 个截面")
    print("width  n_stable@0.001  @0.0003  @0.0001")
    for w in WIDTHS:
        hits = [next((n for n in SIZES if all(scores[w,n,s,b] >= .99 for s in range(3))), None)
                for b in LEVELS]
        print(w, *hits, sep="\t")
    deltas = [scores[2048,n,s,b] - scores[128,n,s,b] for n,s,b in
              itertools.product((n for n in SIZES if 512 <= n <= 1152), range(3), LEVELS)]
    print(f"W2048 > W128: {sum(d > 0 for d in deltas)}/{len(deltas)}; "
          f"median change: {100 * median(deltas):.2f} pp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="可选：原始正式 ZIP，用来核对精简 CSV")
    parser.add_argument("--export-csv", type=Path, help="仅配合 --archive，生成逐次记录 CSV")
    args = parser.parse_args()
    if args.export_csv and not args.archive:
        parser.error("--export-csv 需要 --archive")
    archive_rows = from_archive(args.archive) if args.archive else None
    if args.export_csv:
        summary(archive_rows)
        with args.export_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(archive_rows)
        return
    with (HERE/"matched_crossings.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if archive_rows is not None:
        if rows != [{k: str(r[k]) for k in FIELDS} for r in archive_rows]:
            raise ValueError("公开 CSV 与原始逐次记录不符")
        with ZipFile(args.archive) as z:
            original = list(csv.DictReader(io.StringIO(z.read("matched_loss_transition_thresholds.csv").decode("utf-8-sig"))))
        for r in original:
            w, b = int(r["width"]), float(r["target_raw_bce"])
            if b not in LEVELS:
                continue
            good = [int(x["train_count"]) for x in rows if int(x["width"])==w and
                    float(x["target_raw_bce"])==b and float(x["matched_probe_exact_accuracy"])>=.99]
            if next(n for n in SIZES if good.count(n)==3) != int(r["n_stable"]):
                raise ValueError(f"与原包阈值表不符：{w}, {b}")
    summary(rows)


if __name__ == "__main__":
    main()
