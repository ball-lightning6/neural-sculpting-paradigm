"""只打包复算或复训所需的小文件，不发布优化器和大权重。"""
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/"research/neural_k_framework/assets/rule30_lag"
E11 = Path("research/neural_k_framework/experiments/experiment_11_rule30_gradient_alignment/experiment_rule30_train_val_gradient_alignment.py")
SOURCE = ROOT/".tmp_rule30_gradient_alignment_results_20260819"


def packed(items):
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w") as archive:
        for name,data in sorted(items.items()):
            entry=zipfile.ZipInfo(name,(2026,11,8,0,0,0))
            archive.writestr(entry,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    return out.getvalue()


def main():
    digest=lambda b:hashlib.sha256(b).hexdigest()
    meta=json.loads((HERE/"run/manifest.json").read_text("utf8"))
    assert digest((HERE/"train.py").read_bytes())==meta["train_sha256"]
    assert digest((HERE/"PROTOCOL_ZH.md").read_bytes())==meta["cfg_sha256"]
    e11={k:(SOURCE/k).read_bytes() for k in meta["data_hashes"]}
    assert {k:digest(v) for k,v in e11.items()}==meta["data_hashes"]
    relative=HERE.relative_to(ROOT).as_posix()
    files=("PROTOCOL_ZH.md","RESULTS_ZH.md","train.py","analyze.py","run/trajectory.csv","run/matched_raw_bce.csv",
           "run/paired_by_seed.csv","run/manifest.json","run/complete.json","run/split_1024.npz","run/curves.png")
    items={f"{relative}/{k}":(HERE/k).read_bytes() for k in files}
    items[E11.as_posix()]=(ROOT/E11).read_bytes()
    items["e11_input.zip"]=packed(e11)
    items["README.txt"]=(f"Rule30, E11, n=1024.\n\n"
        f"复算 / Recalculate (Python standard library):\n  python {relative}/analyze.py\n"
        f"重画图 / Plot (Matplotlib):\n  python {relative}/analyze.py --plot\n"
        f"复训 / Retrain (original local protocol: PyTorch, NumPy, Matplotlib, cuda:1):\n"
        f"  python {relative}/train.py --archive e11_input.zip --out new_run\n"
        f"  python {relative}/analyze.py --run new_run\n"
        f"三组共享数据、种子及每步抽样；不含大权重。/ Conditions share data, initializations and batches. No large weights.\n").encode()
    OUT.mkdir(exist_ok=True)
    (OUT/"paired_1024.zip").write_bytes(packed(items))
    shutil.copyfile(HERE/"run/curves.png",OUT/"paired_1024.png")


if __name__=="__main__":
    main()
