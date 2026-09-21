"""分析固定范数球端点网格的增量和交互项。无 argparse。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results_endpoint_uniform_ball"
TARGET_THRESHOLDS = (0.01, 0.003, 0.001)
LOW_VALUES = (-11.0, -10.0, -9.0)
HIGH_VALUES = (9.0, 10.0, 11.0)


def load(target_threshold: float) -> dict[tuple[float, float, str], dict]:
    records = json.loads((RESULT_DIR / "records.json").read_text(encoding="utf-8"))
    result = {}
    for record in records:
        curve = [row for row in record["curve"] if abs(float(row["threshold"]) - target_threshold) < 1e-12]
        if not curve:
            continue
        row = curve[-1]
        case = record["case"]
        result[(float(case["low"]), float(case["high"]), record["function_hex"])] = {
            "k": float(row["neural_k_bits"]),
            "sd": float(row["k_replica_sd_bits"]),
            "status": record["status"],
        }
    return result


def one_grid(k: dict, function_hex: str) -> dict:
    rows = []
    for low in LOW_VALUES[:-1]:
        for high in HIGH_VALUES[:-1]:
            corners = {
                "base": (low, high),
                "low_out": (low - 1.0, high),
                "high_out": (low, high + 1.0),
                "both_out": (low - 1.0, high + 1.0),
            }
            if not all((a, b, function_hex) in k for a, b in corners.values()):
                continue
            values = {name: k[(a, b, function_hex)] for name, (a, b) in corners.items()}
            gamma = values["both_out"]["k"] - values["low_out"]["k"] - values["high_out"]["k"] + values["base"]["k"]
            error_proxy = sum(values[name]["sd"] for name in corners)
            rows.append({
                "function_hex": function_hex,
                "base_low": low,
                "base_high": high,
                "gamma_bits": gamma,
                "replica_sd_sum_proxy": error_proxy,
                "gamma_over_sd_proxy": abs(gamma) / error_proxy if error_proxy else None,
                "delta_low_bits": values["low_out"]["k"] - values["base"]["k"],
                "delta_high_bits": values["high_out"]["k"] - values["base"]["k"],
                "k_base": values["base"]["k"],
                "k_both": values["both_out"]["k"],
            })
    return {"rectangles": rows}


def symmetric_path(k: dict, function_hex: str) -> dict:
    points = []
    for amplitude in (9.0, 10.0, 11.0):
        key = (-amplitude, amplitude, function_hex)
        if key not in k:
            return {"points": [], "second_difference_bits": None}
        points.append({"amplitude": amplitude, **k[key]})
    second = points[2]["k"] - 2.0 * points[1]["k"] + points[0]["k"]
    return {"points": points, "second_difference_bits": second}


def main() -> None:
    summaries = []
    for target_threshold in TARGET_THRESHOLDS:
        data = load(target_threshold)
        functions = sorted({key[2] for key in data})
        rectangles = [one_grid(data, function_hex) for function_hex in functions]
        symmetric = [
            {"function_hex": function_hex, **symmetric_path(data, function_hex)}
            for function_hex in functions
        ]
        summaries.append({
            "threshold": target_threshold,
            "records_at_threshold": len(data),
            "functions": functions,
            "rectangles": rectangles,
            "symmetric_paths": symmetric,
        })
    summary = {
        "measure": "uniform Lebesgue measure in ||theta||_2 <= 72",
        "thresholds": list(TARGET_THRESHOLDS),
        "summaries": summaries,
        "interpretation": {
            "gamma": "K(low-1,high+1)-K(low-1,high)-K(low,high+1)+K(low,high)",
            "small_gamma": "同一窗口内两个端点新增约束的log-volume代价近似可加",
            "error_proxy": "四个replica标准差之和，只作保守诊断，不是置信区间",
        },
    }
    (RESULT_DIR / "grid_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 端点网格分析",
        "",
        "测度：固定 `R=72` 球内均匀欧式测度。",
        "",
        "## 交互项",
        "",
        "`Gamma = K(low-1, high+1) - K(low-1, high) - K(low, high+1) + K(low, high)`。",
        "",
        "| MSE | 规则 | 基准端点 | Gamma(bit) | replica SD和 | |Gamma|/SD和 |"
    ]
    for item in summaries:
        threshold = item["threshold"]
        for block in item["rectangles"]:
            for row in block["rectangles"]:
                lines.append(
                    f"| {threshold:g} | {row['function_hex']} | ({row['base_low']:+g},{row['base_high']:+g}) | "
                    f"{row['gamma_bits']:.4f} | {row['replica_sd_sum_proxy']:.4f} | "
                    f"{row['gamma_over_sd_proxy']:.2f} |"
                )
    lines.extend(["", "## 对称幅度路径", "", "| MSE | 规则 | K(9) | K(10) | K(11) | 二阶差分 |"])
    for item in summaries:
        for row in item["symmetric_paths"]:
            points = row["points"]
            if len(points) == 3:
                lines.append(
                    f"| {item['threshold']:g} | {row['function_hex']} | {points[0]['k']:.4f} | {points[1]['k']:.4f} | "
                    f"{points[2]['k']:.4f} | {row['second_difference_bits']:.4f} |"
                )
    lines.extend([
        "",
        "二阶差分接近0表示这一有限窗口内相邻收缩增量接近恒定；必须结合replica误差和更多窗口判断。",
    ])
    (RESULT_DIR / "GRID_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(data), "functions": functions, "analysis": str(RESULT_DIR / "grid_analysis.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
