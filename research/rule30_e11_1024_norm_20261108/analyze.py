"""独立读取完整训练轨迹，复算配对读数。分析不导入 Torch。"""
import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
ARMS = ("baseline","high_norm","low_norm")
NAMES = ("无惩罚","惩罚大范数","惩罚小范数")
SEEDS = tuple(range(5))


def load(folder):
    meta=json.loads((folder/"manifest.json").read_text("utf8"))
    end=json.loads((folder/"complete.json").read_text("utf8"))
    assert end["signature"]==meta["signature"] and end["steps"]==20000
    with (folder/"trajectory.csv").open(encoding="utf8",newline="") as f:
        runs={(a,s):[] for a in ARMS for s in SEEDS}
        for x in csv.DictReader(f):
            k=(x.pop("arm"),int(x.pop("seed"))); t=int(x.pop("step"))
            r={name: (float(value) if value!="" else None) for name,value in x.items()}
            r.update(step=t,gap=r["val_bce"]-r["train_bce"])
            runs[k].append(r)
    grids={tuple(x["step"] for x in r) for r in runs.values()}
    assert len(grids)==1 and sum(map(len,runs.values()))==end["rows"]
    grid=next(iter(grids))
    assert tuple(sorted(set(grid)))==grid and grid[0]==0 and grid[-1]==20000
    for s in SEEDS:
        r0=[runs[a,s][0] for a in ARMS]
        for k in ("train_bce","val_bce","train_exact","val_exact","radius"):
            assert max(x[k] for x in r0)-min(x[k] for x in r0)<1e-7
    return meta,end,runs


def nearest(rows,level,grad=False):
    pool=[r for r in rows if not grad or r["raw_grad_cos"] is not None]
    r=min(pool,key=lambda x:abs(math.log(x["train_bce"]/level)))
    return r if 1/1.35 <= r["train_bce"]/level <= 1.35 else None


def crossing(rows,target):
    for u,v in zip(rows,rows[1:]):
        x,y=u["train_bce"],v["train_bce"]
        if x>target>=y:
            a=math.log(target/x)/math.log(y/x)
            return dict(step_before=u["step"],step_after=v["step"],bce_before=x,bce_after=y,
                        val_exact_before=u["val_exact"],val_exact_after=v["val_exact"],
                        train_exact_before=u["train_exact"],train_exact_after=v["train_exact"],
                        **{k:(1-a)*u[k]+a*v[k] for k in ("val_exact","val_bce","step")})
    return None


def hits(rows):
    def first(condition):
        return next((r for r in rows if condition(r)),None)
    def step(x):
        return x["step"] if x is not None else None
    train=first(lambda r:r["train_exact"]==1)
    val=first(lambda r:r["val_exact"]==1)
    stable=None
    for r in reversed(rows):
        if r["val_exact"]!=1:
            break
        stable=r
    b1,b2=(first(lambda r:r[k]<=.003) for k in ("train_bce","val_bce"))
    return dict(train_full=step(train),val_first=step(val),val_stable=step(stable),
                lag_first=(step(val)-step(train) if val and train else None),
                lag_stable=(step(stable)-step(train) if stable and train else None),
                stable_train_bce=(stable["train_bce"] if stable else None),
                train003=step(b1),val003=step(b2),lag003=(step(b2)-step(b1) if b1 and b2 else None),
                final_exact=rows[-1]["val_exact"],final_radius=rows[-1]["radius"])


def median_complete(rows,key):
    vals=[r[key] for r in rows]
    return median(vals) if all(v is not None for v in vals) else None


def analyze(folder):
    meta,end,runs=load(folder)
    at={(a,s,r["step"]):r for (a,s),rr in runs.items() for r in rr}
    per=[dict(arm=a,seed=s,**hits(runs[a,s])) for a in ARMS for s in SEEDS]
    bykey={(r["arm"],r["seed"]):r for r in per}
    with (folder/"paired_by_seed.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=per[0]);w.writeheader();w.writerows(per)
    points={}
    matched={}
    for a in ARMS:
        points[a]={}
        for t in (0,20,50,100,500,1000,2000,4000,8000,20000):
            points[a][t]={k:mean(at[a,s,t][k] for s in SEEDS) for k in
                          ("train_bce","val_bce","gap","train_exact","val_exact","radius","penalty")}
        matched[a]={}
        for level in (.1,.01,.003):
            eligible=[nearest(runs[a,s],level) for s in SEEDS]
            gr=[nearest(runs[a,s],level,True) for s in SEEDS]
            eligible=[r for r in eligible if r is not None]; gr=[r for r in gr if r is not None]
            matched[a][level]=dict(n=len(eligible),actual_bce=mean(r["train_bce"] for r in eligible) if eligible else None,
                                   val_bce=mean(r["val_bce"] for r in eligible) if eligible else None,
                                   val_exact=mean(r["val_exact"] for r in eligible) if eligible else None,
                                   n_grad=len(gr),grad_actual_bce=mean(r["train_bce"] for r in gr) if gr else None,
                                   grad_cos=mean(r["raw_grad_cos"] for r in gr) if gr else None)
    levels=(.1,.03,.01,.003,.001)
    crossings=[dict(arm=a,seed=s,target_bce=lev,**crossing(runs[a,s],lev)) for lev in levels for a in ARMS for s in SEEDS]
    with (folder/"matched_raw_bce.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=crossings[0]); w.writeheader(); w.writerows(crossings)
    cx={(r["arm"],r["seed"],r["target_bce"]):r for r in crossings}
    def endpoint_order(target,arm):
        count=0
        for s in SEEDS:
            def bounds(a):
                p=cx[a,s,target]
                return sorted([p["val_exact_before"],p["val_exact_after"]])
            b,a=bounds("baseline"),bounds(arm)
            count+= (a[0]>b[1] if arm=="high_norm" else a[1]<b[0])
        return count
    datafile=folder/"trajectory.csv"
    result=dict(signature=end["signature"],trajectory_sha256=sha256(datafile.read_bytes()).hexdigest(),
                points=points,matched=matched,by_seed=per,crossings=crossings,
                separated_at_003={a:endpoint_order(.003,a) for a in ARMS[1:]})
    (folder/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),"utf8")
    def f(x,fmt):
        return format(x,fmt) if x is not None else "未达"
    def pc(x):
        return f(x*100,".3f")+"%" if x is not None else "未达"
    report=["# E11 原始 1024 个训练样本：范数惩罚有没有缩短验证滞后？","",
       "完成 3 组 × 5 个种子，每次固定 20000 步，没有用验证成绩早停。原 E11 的 1024 条训练输入、4096 条独立验证输入和初始化规则均已找回；旧结果不是本次基线。以下准确率均要求整道 30 位输出全对。","",
       "## 最重要的区别：相同步数不是相同训练损失","",
       "目标函数是原始训练 BCE 加上范数惩罚。即使只看不含惩罚的原始 BCE，三组下降的速度也不同。第 100 步，无惩罚、惩罚大范数、惩罚小范数三组的平均原始训练 BCE 分别为 "
       +"、".join(f"{points[a][100]['train_bce']:.5f}" for a in ARMS)+"。所以不能只按训练步数判断哪种模型更会泛化。","",
       "下表将每个种子对齐到同一原始训练 BCE，再平均其验证整串准确率。取首次向下穿过目标的前后两个实测点，在对数训练 BCE 上插值。这是检查点间的估计，不是实测了恰好等于目标的权重。","",
       "| 目标原始训练 BCE | 无惩罚 | 惩罚大范数 | 惩罚小范数 |","|---:|---:|---:|---:|"]
    for lev in levels:
        report.append("| "+str(lev)+" | "+" | ".join(pc(mean(cx[a,s,lev]["val_exact"] for s in SEEDS)) for a in ARMS)+" |")
    report+=["",f"在 0.003 处，三组所有种子在跨阈值的前后两个节点都已训练整串全对。即使不用插值，{endpoint_order(.003,'high_norm')}/5 个种子的高范数惩罚组前后两次验证准确率都高于基线的两次，{endpoint_order(.003,'low_norm')}/5 个种子的低范数惩罚组两次都低于基线的两次。"
             "这说明时间差不只是 BCE 下降快慢；同一平均 BCE 仍可能由不同的逐样本 margin、隐藏表示和训练路径达到，不能凭这一比较把差异唯一归因于静态简单性偏置。","",
       "[每个种子的跨阈值前后读数与插值](run/matched_raw_bce.csv)。以下再看训练过程及达到零错所需的时间。","",
       "## 在相同步数，验证集学到什么程度？","",
       "| 条件 | 第 50 步 BCE 差 | 第 100 步 BCE 差 | 第 100 步训练全对率 | 第 100 步验证全对率 | 第 500 步验证全对率 | 终点验证零错种子 |",
       "|---|---:|---:|---:|---:|---:|---:|"]
    for a,name in zip(ARMS,NAMES):
        d=points[a]; count=sum(bykey[a,s]["final_exact"]==1 for s in SEEDS)
        report.append(f"| {name} | {d[50]['gap']:.5f} | {d[100]['gap']:.5f} | {pc(d[100]['train_exact'])} | {pc(d[100]['val_exact'])} | {pc(d[500]['val_exact'])} | {count}/5 |")
    report+=["","差值为同一时刻验证原始 BCE 减训练原始 BCE，不加范数项；不能把它当成达到同一损失所需的步数差。",
             "","## 训练集全对后，验证集还要等多久？","",
             "下表是每组的三个数：训练集首次全对 / 验证集首次全对 / 验证集全对且其后每个记录都保持全对的起始步数。未达者保留，不用预算上限代替。","",
             "| 种子 | 无惩罚 | 惩罚大范数 | 惩罚小范数 |","|---:|---|---|---|"]
    for s in SEEDS:
        report.append("| "+str(s)+" | "+" | ".join(" / ".join(f(bykey[a,s][k],"d") for k in ("train_full","val_first","val_stable")) for a in ARMS)+" |")
    report+=["","| 条件 | 稳定验证全对的中位等待步数 | 验证与训练首次达到原始 BCE 0.003 的中位步数差 | 稳定验证全对时的中位训练 BCE |",
             "|---|---:|---:|---:|"]
    for a,name in zip(ARMS,NAMES):
        rr=[bykey[a,s] for s in SEEDS]
        report.append("| "+name+" | "+" | ".join(f(median_complete(rr,k),".5g") for k in ("lag_stable","lag003","stable_train_bce"))+" |")
    report+=["","仅当 5 个种子全部达标时才给出未删失的中位等待。步数都是首次保存的达标节点，不插值；早期每 5 步、200 至 1000 步后段每 25 步、1000 至 4000 步每 50 步，后续逐步放宽。","",
             "## 在相近的训练损失处比较","",
             "| 目标训练 BCE | 条件 | 命中数 | 实际训练 BCE 均值 | 验证 BCE 均值 | 验证整串准确率 | 梯度命中数 | 梯度节点实际训练 BCE | 原始训练/验证梯度余弦 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for level in (.1,.01,.003):
        for a,name in zip(ARMS,NAMES):
            r=matched[a][level]
            report.append("| "+" | ".join([str(level),name,str(r["n"]),f(r["actual_bce"],".5g"),f(r["val_bce"],".5g"),pc(r["val_exact"]),
                str(r["n_grad"]),f(r["grad_actual_bce"],".5g"),f(r["grad_cos"],".4f")])+" |")
    report+=["","每次训练各自在记录中找对数距离最近的节点，实际训练 BCE 须在目标的 1/1.35 至 1.35 倍以内。梯度只在较密的早期指定节点测量，因此和左侧读数不一定是同一步；未命中不是梯度为零。","",
             "原始 [逐步记录](run/trajectory.csv)、[逐种子穿越记录](run/paired_by_seed.csv)、[数据与运行签名](run/manifest.json)、[训练前方案](PROTOCOL_ZH.md)。本组是共享一份划分的训练干预，不是对静态体积或普适最优范数的测量。"]
    (folder.parent/"RESULTS_ZH.md").write_text("\n".join(report)+"\n",encoding="utf8")
    return runs


def plot(runs,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13.5,4),layout="constrained")
    for a,label,color in zip(ARMS,("No penalty","Penalize large norms","Penalize small norms"),("#485569","#12735e","#be5b32")):
        rr=[runs[a,s] for s in SEEDS]
        t=[r["step"] for r in rr[0]]
        for ax in axes:
            ax.grid(axis="y",alpha=.14)
        g=[mean(row[j]["gap"] for row in rr) for j in range(len(t))]
        acc=[100*mean(row[j]["val_exact"] for row in rr) for j in range(len(t))]
        axes[0].plot(t,g,color=color,label=label)
        axes[1].plot(t,acc,color=color,label=label)
        levels=[10**(-1-j/28) for j in range(85)]
        aligned=[100*mean(crossing(row,v)["val_exact"] for row in rr) for v in levels]
        axes[2].plot(levels,aligned,color=color,label=label)
    axes[0].set(xlim=(0,200),ylabel="Validation BCE minus training BCE",title="A. Early loss gap")
    axes[1].set_xscale("log")
    axes[1].set(xlim=(20,4000),ylim=(-2,102),ylabel="Whole-ring validation accuracy (%)",title="B. Generalization following training fit")
    axes[1].set_xticks([20,50,100,500,1000,4000],["20","50","100","500","1k","4k"])
    axes[2].set_xscale("log")
    axes[2].set(xlim=(.1,.0001),ylim=(-2,102),ylabel="Whole-ring validation accuracy (%)",
                title="C. At matched raw training BCE")
    axes[2].set_xticks([.1,.01,.001,.0001],["0.1","0.01","0.001","0.0001"])
    for ax in axes:
        ax.set_xlabel("Training updates");ax.spines[["top","right"]].set_visible(False)
    axes[2].set_xlabel("Training BCE (lower to the right)")
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(path,dpi=160);plt.close(fig)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--run",type=Path,default=HERE/"run")
    ap.add_argument("--plot",action="store_true");args=ap.parse_args()
    records=analyze(args.run)
    if args.plot:
        plot(records,args.run/"curves.png")
    print(args.run.parent/"RESULTS_ZH.md")
