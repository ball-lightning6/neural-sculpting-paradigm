"""在原始 E11 的 1024 个样本上逐种子配对，训练过程中不据验证集选模型。"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "research/neural_k_framework/experiments/experiment_11_rule30_gradient_alignment/experiment_rule30_train_val_gradient_alignment.py"
LEGACY = ROOT / ".tmp_rule30_gradient_alignment_results_20260819"
ARMS = ("baseline", "high_norm", "low_norm")
SEEDS = (0, 1, 2, 3, 4)
NMODEL = len(SEEDS) * len(ARMS)
STEPS = 20000
SIZES = (256, 512, 768, 1024, 1280, 1536, 2048, 4096)
R2_HIGH, R2_LOW = 1.0915566140657849, 3.399311254876147
CHECKS = sorted({0, 1, 2, 5, 71, 84, 91, *range(10, 201, 5), *range(225, 1001, 25),
                 *range(1050, 4001, 50), *range(4250, 8001, 250), *range(8500, STEPS + 1, 500)})
GRAD_STEPS = set(range(40, 141, 5)) | {150, 175, 200}
HASHES = {"dataset_split_ids.npz":"cff5ead89715e601e5e5bf592dd46be7730ec73ca7f50169c6206b0048d43f47",
          "config.json":"e6fe9693bbea162ac33b5bb41bda56d448c8d600aaf6f94003d8bfd2df511c5a",
          "trajectory.csv":"72b0b2192dd1745598233f0435415396b3fd6b33d5874d91fe39186e8415779c"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    os.replace(tmp, path)


def dataset(archive, original, dev, out):
    if archive is None and LEGACY.is_dir():
        blobs = {k: (LEGACY/k).read_bytes() for k in HASHES}
    else:
        if archive is None:
            archive = Path.home() / "Downloads/results_rule30_train_val_gradient_alignment_package.zip"
        with ZipFile(archive) as z:
            blobs = {k: z.read(k) for k in HASHES}
    if {k:sha(b) for k,b in blobs.items()} != HASHES:
        raise RuntimeError("E11 的划分、配置或轨迹哈希不匹配")
    cfg = original.build_runtime_config()
    assert tuple(cfg.train_counts) == SIZES and cfg.train_order_seed == 314159 and cfg.data_seed == 20260819
    assert sha(SOURCE.read_bytes()) == "811ff2b3fb8b6bc16431b8226c02540633591be1b75c98b16948f6933bde00df"
    regenerated = original.make_nested_dataset(cfg)
    with np.load(io.BytesIO(blobs["dataset_split_ids.npz"]), allow_pickle=False) as ids:
        for k in ("train_ids", "validation_ids"):
            assert np.array_equal(ids[k], regenerated[k])
        assert not np.intersect1d(ids["train_ids"][:1024],ids["validation_ids"]).size
        np.savez_compressed(out/"split_1024.npz", train_ids=ids["train_ids"][:1024], validation_ids=ids["validation_ids"])
    data = [torch.from_numpy(regenerated[k][:end]).to(dev) for k,end in
            (("train_x",1024),("train_y",1024),("validation_x",4096),("validation_y",4096))]
    old = {int(row["model_seed"]): row for row in csv.DictReader(io.StringIO(blobs["trajectory.csv"].decode("utf-8-sig")))
           if int(row["train_count"]) == 1024 and int(row["step"]) == 0}
    return cfg,data,old


def batch_plan(cfg, original, device):
    generator = torch.Generator(device=device).manual_seed(cfg.train_order_seed)
    sampled = [n for n in cfg.train_counts if n > cfg.batch_size]
    def draw():
        for n in sampled:
            a = torch.randint(n,(cfg.batch_size,),device=device,generator=generator)
            if n == 1024:
                target = a
        return target
    plan = torch.empty((STEPS,cfg.batch_size),dtype=torch.long,device=device)
    for start in range(0,STEPS,100):
        plan[start:start+100].copy_(torch.stack([draw() for _ in range(min(100,STEPS-start))]))
    other = torch.Generator(device=device).manual_seed(cfg.train_order_seed)
    ns = torch.tensor(cfg.train_counts,device=device)
    for step in range(4):
        indices,_ = original.make_training_batch(cfg,ns,len(SIZES),5,other)
        assert torch.equal(plan[step],indices[SIZES.index(1024)*5])
    return plan


def radius2(model):
    layers = list(zip(model.hidden_weights,model.hidden_biases)) + [(model.output_weight,model.output_bias)]
    return sum(3*w.shape[-1]*(w.square().sum((1,2))+b.square().sum(1)) for w,b in layers) / sum(
        w[0].numel()+b[0].numel() for w,b in layers)


def penalty(r2, masks):
    return .1*(masks[0]*F.relu(r2/R2_HIGH-1).square() + masks[1]*F.relu(1-r2/R2_LOW).square())


def batched(x):
    return x.unsqueeze(0).expand(NMODEL,-1,-1)


def capture(model, opt, x, y, masks, dev):
    before = [p.detach().clone() for p in model.parameters()]
    def train_step():
        opt.zero_grad(set_to_none=True)
        per_point = F.binary_cross_entropy_with_logits(model(batched(x)),batched(y),reduction="none").mean(2)
        (per_point.mean(1)+penalty(radius2(model),masks)).sum().backward()
        opt.step()
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            train_step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        train_step()
    torch.cuda.synchronize(dev)
    with torch.no_grad():
        for p,init in zip(model.parameters(),before):
            p.copy_(init)
            for val in opt.state[p].values():
                if torch.is_tensor(val):
                    val.zero_()
    return graph


@torch.no_grad()
def metrics(model, x, y):
    sums = x.new_zeros((NMODEL,3),dtype=torch.float64)
    for a,b in zip(x.split(256),y.split(256)):
        z = model(batched(a))
        loss = F.binary_cross_entropy_with_logits(z,batched(b),reduction="none")
        correct = (z>=0)==(b>=.5).unsqueeze(0)
        sums[:,0] += loss.sum((1,2),dtype=torch.float64)
        sums[:,1] += correct.all(dim=2).sum(1)
        sums[:,2] += correct.sum((1,2))
    return (sums / sums.new_tensor([y.numel(),len(y),y.numel()])).cpu().numpy()


def full_grad(model, x, y):
    params = tuple(model.parameters())
    total = [torch.zeros_like(p) for p in params]
    for a,b in zip(x.split(256),y.split(256)):
        z = model(batched(a))
        obj = F.binary_cross_entropy_with_logits(z,batched(b),reduction="sum") / y.numel()
        for p,g in zip(total,torch.autograd.grad(obj,params)):
            p.add_(g)
    return total


def grad_cosine(model,data):
    first = full_grad(model,*data[:2])
    other = full_grad(model,*data[2:])
    def dot(a,b):
        return sum((x*y).flatten(1).sum(1) for x,y in zip(a,b))
    return (dot(first,other)/(dot(first,first)*dot(other,other)).sqrt()).cpu().tolist()


def save_state(path, model, opt, step, rows, sig):
    tmp = path.with_suffix(".part")
    torch.save((step,model.state_dict(),opt.state_dict(),rows,sig),tmp)
    os.replace(tmp,path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive",type=Path)
    ap.add_argument("--out",type=Path,default=HERE/"run")
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    lock = out/"active.lock"
    with lock.open("x",encoding="ascii") as handle:
        handle.write(str(os.getpid()))
    try:
        train(args.archive,out)
    finally:
        lock.unlink()


def train(archive,out):
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError("需要本机 CUDA1，不回退到 CPU 或显示卡")
    dev = torch.device("cuda:1"); torch.cuda.set_device(dev)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    spec = importlib.util.spec_from_file_location("e11_original",SOURCE)
    original = importlib.util.module_from_spec(spec); sys.modules[spec.name]=original; spec.loader.exec_module(original)
    cfg,data,old = dataset(archive,original,dev,out)
    labels = [(a,s) for a in ARMS for s in SEEDS]
    masks = tuple(torch.tensor([a==which for a,s in labels],dtype=torch.float32,device=dev) for which in ARMS[1:])
    model = original.BatchedIndependentMLP(30,30,1024,3,[s for a,s in labels],"gelu",1e-5).to(dev)
    for p in model.parameters():
        assert torch.equal(p[:5],p[5:10]) and torch.equal(p[:5],p[10:])
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.,foreach=False,capturable=True)
    plan = batch_plan(cfg,original,dev)
    meta = dict(data_hashes=HASHES, batch_sha256=sha(plan.cpu().numpy().astype("<u2").tobytes()),
                cfg_sha256=sha((HERE/"PROTOCOL_ZH.md").read_bytes()), train_sha256=sha(Path(__file__).read_bytes()),
                device=torch.cuda.get_device_name(dev),torch=str(torch.__version__),cuda=torch.version.cuda,
                dtype="float32",tf32=True,steps=STEPS,seeds=SEEDS,arms=ARMS,grad_steps=sorted(GRAD_STEPS),
                train=1024,validation=4096,lambda_=0.1,high_cutoff2=R2_HIGH,low_cutoff2=R2_LOW)
    sig = sha(json.dumps(meta,sort_keys=True).encode())
    if (out/"manifest.json").is_file():
        assert json.loads((out/"manifest.json").read_text("utf8"))["signature"]==sig, "条件改变，不覆写旧运行"
    write_json(out/"manifest.json",{**meta,"signature":sig})
    if (out/"complete.json").exists():
        print("已完成；不重复训练。",flush=True); return
    x=data[0][plan[0]].clone(); y=data[1][plan[0]].clone()
    graph=capture(model,opt,x,y,masks,dev)
    rows=[]; start=0; ckpt=out/"resume.pt"
    if ckpt.exists():
        start,params,state,rows,saved_sig=torch.load(ckpt,map_location=dev,weights_only=True)
        assert saved_sig==sig
        model.load_state_dict(params)
        for group,st in zip(opt.param_groups[0]["params"],state["param_groups"][0]["params"]):
            for k,v in state["state"][st].items():
                opt.state[group][k].copy_(v)
        del params,state
    print(f"CUDA1={meta['device']}; 15 个配对模型；开始步数={start}",flush=True)
    began=time.perf_counter()
    selected=set(CHECKS)
    for t in range(start,STEPS+1):
        if t>start:
            x.copy_(data[0][plan[t-1]])
            y.copy_(data[1][plan[t-1]])
            graph.replay()
        if t not in selected or (start and t==start):
            continue
        tr,va=metrics(model,*data[:2]),metrics(model,*data[2:])
        cos=grad_cosine(model,data) if t in GRAD_STEPS else [None]*NMODEL
        with torch.no_grad():
            r=radius2(model); pen=penalty(r,masks).cpu().tolist(); rr=r.sqrt().cpu().tolist()
        new=[]
        for i,(arm,s) in enumerate(labels):
            row=dict(arm=arm,seed=s,step=t,train_bce=float(tr[i,0]),val_bce=float(va[i,0]),
                     train_exact=float(tr[i,1]),val_exact=float(va[i,1]),train_bit=float(tr[i,2]),val_bit=float(va[i,2]),
                     radius=rr[i],penalty=pen[i],raw_grad_cos=cos[i])
            if not all(math.isfinite(v) for v in (row["train_bce"],row["val_bce"],rr[i],pen[i])) or (cos[i] is not None and not math.isfinite(cos[i])):
                raise RuntimeError(f"非有限记录：{row}")
            if t==0 and max(abs(row[k]-float(old[s][j])) for k,j in
                            (("train_bce","train_raw_bce"),("val_bce","validation_raw_bce")))>0.00025:
                raise RuntimeError("同种子的初始模型和旧 E11 读数不符")
            new.append(row)
        rows.extend(new)
        if t<=200 or t%250==0 or t==CHECKS[-1]:
            write_json(out/"progress.json",{"step":t,"seconds":round(time.perf_counter()-began,2),"rows":new})
        if t in (0,50,100,500,1000) or t%1000==0:
            vals=[float(np.mean(va[i:i+5,1])) for i in (0,5,10)]
            print(f"step={t}, val_exact={vals}, elapsed={time.perf_counter()-began:.1f}s",flush=True)
        if t==500 or (t and t%2000==0):
            save_state(ckpt,model,opt,t,rows,sig)
            with (out/"trajectory.csv").open("w",encoding="utf8",newline="") as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    assert len(rows)==len(CHECKS)*NMODEL
    write_json(out/"complete.json",{"signature":sig,"steps":STEPS,"rows":len(rows),"last_session_seconds":time.perf_counter()-began,
                                    "peak_reserved_gib":torch.cuda.max_memory_reserved(dev)/2**30})
    print("训练完成。",flush=True)


if __name__=="__main__":
    main()
