"""从保存的 Rule30 检查点复算训练/验证滞后，并单独核对 E11 梯度。不会训练网络。"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
ARMS = ("baseline", "high_norm", "low_norm")
SIZES = (600, 750, 900, 1200)
STEPS = (0, 1, 10, 100, *range(500, 20001, 500))
E11_SIZES = (256, 512, 768, 1024, 1280, 1536, 2048, 4096)
H_FIELDS = ("arm", "n", "seed", "step", "train_bce", "probe_bce", "train_exact", "probe_exact")
G_FIELDS = ("n", "seed", "step", "train_bce", "validation_bce", "gradient_cosine")
E_FIELDS = ("n", "seed", "step", "train_bce", "validation_bce",
            "train_exact", "validation_exact", "validation_bit")
ORIGINAL_HASHES = (
    "6f0392fe9071c7d59373319129bf9b785b2c27e0f410181013b3437f4bb622ae",
    "99e903329941c2946f2cace390221ed76a14af4e16b3440eb6a5961ecd0616fc",
    "dd0f2b0dbf3d5b829a23199edda5507335885d79ff5197e3860f4d01f0ca702b",
)


def extract(paths):
    contents = [p.read_bytes() for p in paths]
    if tuple(hashlib.sha256(b).hexdigest() for b in contents) != ORIGINAL_HASHES:
        raise ValueError("来源不是本次使用的两份历史轨迹与 E11 归档")
    sides = []
    for data, treatment in zip(contents[:2], ARMS[1:]):
        side = {}
        for checkpoint in json.loads(data):
            for r in checkpoint:
                key = (r["loss"], r["n"], r["seed"], r["step"])
                if key in side or key[0] not in ("baseline", treatment):
                    raise ValueError(f"重复或意外的轨迹记录：{key}")
                side[key] = tuple(r[k] for k in H_FIELDS[4:])
        sides.append(side)
    # 两次范数实验都跑了基线；它们的轨迹相同，不能算作八个种子。
    baselines = [{k:v for k,v in side.items() if k[0] == "baseline"} for side in sides]
    if not baselines[0] or baselines[0] != baselines[1]:
        raise ValueError("两份基线的记录不一致")
    histories = [(*k, *v) for k,v in sorted((sides[0] | sides[1]).items())]
    with ZipFile(io.BytesIO(contents[2])) as archive:
        with archive.open("fixed_raw_bce_comparison.csv") as data:
            rows = list(csv.DictReader(io.TextIOWrapper(data, encoding="utf-8-sig")))
        with archive.open("trajectory.csv") as data:
            e11 = [(int(r["train_count"]), int(r["model_seed"]), int(r["step"]),
                    *(float(r[k]) for k in ("train_raw_bce","validation_raw_bce","train_exact_accuracy",
                                             "validation_exact_accuracy","validation_bit_accuracy")))
                   for r in csv.DictReader(io.TextIOWrapper(data, encoding="utf-8-sig"))]
    gradients = []
    for r in rows:
        if float(r["target_train_raw_bce"]) != .1:
            continue
        n, train = int(r["train_count"]), float(r["train_raw_bce"])
        if max(train / .1, .1 / train) > 1.35:
            continue
        if int(r["train_gradient_sample_count"]) != n or int(r["validation_gradient_sample_count"]) != 4096:
            raise ValueError("不是预期的完整梯度")
        gradients.append((n, int(r["model_seed"]), int(r["step"]), train,
                          float(r["validation_raw_bce"]), float(r["train_validation_gradient_cosine"])))
    return histories, sorted(gradients), sorted(e11)


def read_csv(name, fields, types):
    with (HERE/name).open(encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        if tuple(next(r)) != fields:
            raise ValueError(f"列名错误：{name}")
        return [tuple(t(x) for t,x in zip(types,row,strict=True)) for row in r]


def validate(histories, gradients, e11):
    h = {r[:4]:r[4:] for r in histories}
    if len(h) != len(histories) or set(h) != set(itertools.product(ARMS,SIZES,range(4),STEPS)):
        raise ValueError("范数轨迹不全或条件重复")
    for train, probe, tr_acc, pr_acc in h.values():
        if not all(math.isfinite(x) and x >= 0 for x in (train,probe)) or not (0 <= tr_acc <= 1 and 0 <= pr_acc <= 1):
            raise ValueError("BCE 或整串准确率不合法")
    keys = {r[:2] for r in gradients}
    if len(keys) != len(gradients) or keys != set(itertools.product(E11_SIZES,range(5))):
        raise ValueError("E11 数据不全或条件重复")
    for _,_,step,train,val,cos in gradients:
        if step < 0 or not (.1/1.35 <= train <= .1*1.35 and math.isfinite(val) and val>=0 and -1 <= cos <= 1):
            raise ValueError("E11 匹配条件、BCE 或梯度余弦不合法")
    e = {r[:3]:r[3:] for r in e11}
    groups = set(itertools.product(E11_SIZES,range(5)))
    if (len(e) != len(e11) or len(e) != 1100 or {k[:2] for k in e} != groups
            or any((n,s,t) not in e for n,s in groups for t in (0,10,20,50,100,200,500,4000,20000))):
        raise ValueError("E11 轨迹不全或重复")
    for b1,b2,at,av,ab in e.values():
        if not all(math.isfinite(b) and b>=0 for b in (b1,b2)) or not all(0<=a<=1 for a in (at,av,ab)):
            raise ValueError("E11 轨迹数据不合法")
    for n,seed,step,train,val,_ in gradients:
        if e.get((n,seed,step), ())[:2] != (train,val):
            raise ValueError(f"梯度读数未匹配到同一种子、同一步的两项损失：{(n,seed,step)}")
    return h,e


def first_hit(h, arm, n, seed, level, metric):
    return next((t for t in STEPS if h[arm,n,seed,t][metric] <= level), None)


def common_steps(e, n):
    return sorted(set.intersection(*[{t for size,seed,t in e if size==n and seed==s} for s in range(5)]))


def e11_mean(e, n, t, metric):
    return mean(e[n,s,t][metric] for s in range(5))


def report(h, gradients, e):
    print("E11: n / final 100% on 4096 whole outputs (out of 5) / first saved train & validation 100%")
    for n in E11_SIZES:
        def full_steps(metric):
            return [min((t for size,s,t in e if (size,s)==(n,seed) and e[n,s,t][metric]==1), default=None)
                    for seed in range(5)]
        print(n, sum(e[n,s,20000][3]==1 for s in range(5)), full_steps(2), full_steps(3))
    print("768: final mean per-bit and whole-string validation accuracy (%)",
          *(f"{100*e11_mean(e,768,20000,i):.5f}" for i in (4,3)))
    def gap(seed,step):
        train,val = e[1024,seed,step][:2]
        return val-train
    print("1024: seeds with a larger BCE gap at step 50 than at step 20:",
          sum(gap(seed,50)>gap(seed,20) for seed in range(5)), "of 5")
    print("E11: n / step / mean train, validation, difference / train, validation whole-string accuracy")
    for n,t in itertools.product((1024,4096),(0,20,50,100,500,4000,20000)):
        a,b,c,d = [e11_mean(e,n,t,i) for i in range(4)]
        print(n,t,*(f"{v:.8g}" for v in (a,b,b-a,c,d)))
    print("E11: n / mean train BCE / mean gradient cosine / mean validation minus train BCE")
    for n in E11_SIZES:
        rows = [r for r in gradients if r[0] == n]
        print(n, *(f"{v:.5f}" for v in (mean(r[3] for r in rows), mean(r[5] for r in rows),
                                         mean(r[4]-r[3] for r in rows))))
    print("\nn / BCE / arm / train first saved steps / probe first saved steps / median paired lag")
    for n,level,arm in itertools.product(SIZES, (.01,.003), ARMS):
        train = [first_hit(h,arm,n,s,level,0) for s in range(4)]
        probe = [first_hit(h,arm,n,s,level,1) for s in range(4)]
        # 表格仅报四组配对全部命中的中位数；未命中保留为删失。
        lag = median(p-t for p,t in zip(probe,train)) if None not in probe+train else "not reported"
        print(n, level, arm, train, probe, lag)
    print("Norm interventions: n / arm / number attaining 100% at final probe / first train & probe 100%")
    for n,arm in itertools.product(SIZES,ARMS):
        hits = [[next((t for t in STEPS if h[arm,n,s,t][i]==1),None) for s in range(4)] for i in (2,3)]
        print(n,arm,sum(h[arm,n,s,20000][3]==1 for s in range(4)),*hits)


def plot(e, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11.3,4.0),layout="constrained")
    steps = [t for t in common_steps(e,1024) if t<=4000]
    for metric,style,color,label in ((0,"--","#36516b","Training"),(1,"-","#b05236","Validation")):
        axes[0].plot(steps,[e11_mean(e,1024,t,metric) for t in steps],style,color=color,lw=2,label=label)
    axes[0].set(yscale="log",ylabel="Raw BCE")
    axes[0].set_xscale("symlog",linthresh=100)
    axes[0].set_xticks([0,50,100,500,2000,4000],["0","50","100","500","2k","4k"])
    axes[0].set_xlim(0,4100)
    for n,mark,color in ((1024,"o","#b05236"),(4096,"s","#177768")):
        steps = [t for t in common_steps(e,n) if t<=500]
        gap = [e11_mean(e,n,t,1)-e11_mean(e,n,t,0) for t in steps]
        axes[1].plot(steps,gap,color=color,lw=2,marker=mark,ms=3,label=f"{n} inputs")
    axes[1].set(ylabel="Validation BCE minus training BCE",xlim=(0,502))
    axes[1].axhline(0,color="#555555",ls=":",lw=1,zorder=0)
    for ax,title in zip(axes,("A. Both losses fall (1024 inputs)","B. The early gap opens, then closes")):
        ax.set_xlabel("Training updates")
        ax.set_title(title,loc="left",fontsize=12)
        ax.grid(axis="y",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(loc="upper right",fontsize=9,frameon=False)
    fig.savefig(output,dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs=3, metavar=("HIGH_HISTORY","LOW_HISTORY","E11_ZIP"))
    parser.add_argument("--write-csv", action="store_true", help="从指定的三份原件生成精简表")
    parser.add_argument("--plot", type=Path, metavar="PNG", help="另存 E11 五种子平均轨迹图，需要 Matplotlib")
    args = parser.parse_args()
    if args.write_csv and not args.sources:
        parser.error("--write-csv 需要 --sources")
    files = (("checkpoints.csv", H_FIELDS, (str,int,int,int,float,float,float,float)),
             ("e11_gradients.csv", G_FIELDS, (int,int,int,float,float,float)),
             ("e11_trajectory.csv", E_FIELDS, (int,int,int,float,float,float,float,float)))
    original = extract(args.sources) if args.sources else None
    if args.write_csv:
        validate(*original)
        for (name,fields,_),rows in zip(files,original):
            with (HERE/name).open("w",encoding="utf-8",newline="") as f:
                writer = csv.writer(f,lineterminator="\n")
                writer.writerow(fields)
                writer.writerows(rows)
    histories, gradients, e11 = (read_csv(*f) for f in files)
    if original is not None and (histories,gradients,e11) != original:
        raise ValueError("公开表与原始记录不一致")
    h,e = validate(histories,gradients,e11)
    report(h,gradients,e)
    if args.plot:
        plot(e,args.plot)


if __name__ == "__main__":
    main()
