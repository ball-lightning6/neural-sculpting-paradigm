"""CPU排查小随机训练集的系统性捷径；不启动训练或SMC。"""

import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.optimize import linprog,OptimizeWarning
from scipy.stats import rankdata
from config import WORK_DIR, EVIDENCE, LOSS_PAIRING_DIR

ROOT=WORK_DIR / "structure"
COUNTS=(1,2,4,8,16,32,48,64,80,96,112,128,256)
CONFIG={'reference_tables':128,'tree_and_local_reference_tables':32,'counts':list(COUNTS),
        'original_seed':202609112301,'reference_seed':202610261701,'input_bits':8,
        'wall_limit_seconds':180.,'scope':'原四表+独立随机表；只诊断，不筛选更合意的数据替代旧实验。'}
IDS=np.arange(256,dtype=np.uint32)
X=((IDS[:,None]>>np.arange(7,-1,-1))&1).astype(np.float32)
MASKS=np.arange(256,dtype=np.uint16)
POPCOUNT=np.array([int(i).bit_count() for i in MASKS])


def mask_int(values):return int.from_bytes(np.packbits(np.asarray(values,dtype=np.uint8),bitorder='little').tobytes(),'little')
BIT_MASKS=[mask_int((IDS>>bit)&1) for bit in range(8)]
PARITIES=[]
for subset in sorted(range(256),key=lambda x:(int(x).bit_count(),x)):
    bits=np.array([int(int(i)&subset).bit_count()%2 for i in IDS],dtype=np.uint8)
    for flip in (0,1):PARITIES.append((subset,int(subset).bit_count(),mask_int(bits^flip)))


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def data(seed):
    rng=np.random.default_rng(seed)
    order=rng.permutation(256).astype(np.int64)
    labels=rng.integers(0,2,256,dtype=np.uint8)
    return labels,order


def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()


def parity_fit(labels,indices):
    selected=mask_int(np.isin(IDS,indices));target=mask_int(labels)
    for subset,degree,pred in PARITIES:
        if not ((pred^target)&selected):return degree,subset
    return None,None


def linear_fit(labels,indices):
    y=2*labels[indices].astype(float)-1
    design=np.column_stack([X[indices],np.ones(len(indices))])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected.*',category=OptimizeWarning)
        result=linprog(np.zeros(9),A_ub=-y[:,None]*design,b_ub=-np.ones(len(indices)),
                       bounds=[(None,None)]*9,method='highs',options={'time_limit':1.,'threads':1})
    if result.status==2:return {'separable':False,'solver_status':2}
    if not result.success:return {'separable':None,'solver_status':int(result.status),'message':str(result.message)}
    margin=y*(design@result.x)
    if margin.min()<1-1e-7:raise RuntimeError('LP可行性残差超过容差')
    return {'separable':True,'solver_status':0,'witness_weights':result.x.tolist(),'min_margin':float(margin.min()),
            'witness_norm_not_minimum':float(np.linalg.norm(result.x))}


def minimum_tree_leaves(labels,indices):
    selected=mask_int(np.isin(IDS,indices));positive=mask_int(labels)&selected
    @lru_cache(None)
    def solve(items):
        p=items&positive
        if not p or p==items:return 1
        best=items.bit_count()
        for mask in BIT_MASKS:
            hi=items&mask;lo=items^hi
            if not hi or not lo:continue
            candidate=solve(lo)+solve(hi)
            best=min(best,candidate)
            if best==2:break
        return best
    return solve(selected)


def local_cubes(labels,indices):
    ids=IDS[indices];y=labels[indices]
    difference=ids[:,None]^ids[None,:]
    matches=(difference[:,None,:]&MASKS[None,:,None])==0
    same=y[:,None]==y[None,:]
    pure=~np.any(matches&(~same[:,None,:]),axis=2)
    counts=matches.sum(2)
    minimum=np.where(pure,POPCOUNT[None],99).min(1)
    eligible=pure&(POPCOUNT[None]==minimum[:,None])
    best=np.argmax(np.where(eligible,counts,-1),axis=1)
    coverage=counts[np.arange(len(ids)),best]
    return {'minimum_literals':minimum.tolist(),'witness_masks':best.tolist(),
            'same_label_training_coverage':coverage.tolist(),
            'median_minimum_literals':float(np.median(minimum)),
            'mean_minimum_literals':float(np.mean(minimum)),
            'median_training_coverage':float(np.median(coverage)),
            'fraction_sharing_with_another_training_sample':float(np.mean(coverage>=2)),
            'median_full_domain_cube_size':float(np.median(np.exp2(8-minimum))),
            'scope':'每点选择最少字面量且不含已见异标签的子立方体，同长度取最大同类覆盖。未见点标签不参与选择。'}


def corr(a,b):
    a=rankdata(a).astype(float);b=rankdata(b).astype(float)
    a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den else None


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    atomic_json(ROOT/'PLAN.json',{'config':CONFIG,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 'methods':['常数/单bit/affine parity严格训练拟合','实权值仿射阈值LP可行性','最少叶节点的精确训练决策树','逐样本纯标签子立方体'],
                                 'limits':'存在短结构插值不等于压缩n个标签，更不等于能预测新的随机标签；这些指标不是Neural K。'})
    start=time.monotonic();rows=[];original_details=[];paired=[]
    existing={'datasets':json.loads((EVIDENCE/'datasets.json').read_text(encoding='utf-8'))}
    tables=[('original',i,CONFIG['original_seed']+1_000_003*i) for i in range(4)]
    tables += [('reference',i,CONFIG['reference_seed']+1_000_003*i) for i in range(CONFIG['reference_tables'])]
    for role,dataset,seed in tables:
        if time.monotonic()-start>CONFIG['wall_limit_seconds']:break
        labels,order=data(seed)
        if role=='original' and digest(X,labels,order)!=existing['datasets'][str(dataset)]['raw_dataset_sha256']:
            raise RuntimeError('未复现原随机表')
        known_infeasible=False
        for n in COUNTS:
            indices=order[:n]
            k,subset=parity_fit(labels,indices)
            lp={'separable':False,'solver_status':2,'nested_infeasibility':True} if known_infeasible else linear_fit(labels,indices)
            if lp['separable'] is False:known_infeasible=True
            row={'role':role,'dataset':dataset,'n':n,'ones':int(labels[indices].sum()),
                 'constant_or_literal_fit':k is not None and k<=1,'parity_fit':k is not None,
                 'minimum_parity_order':k,'linear_separable':lp['separable'],'LP_status':lp['solver_status']}
            if role=='original' or dataset<CONFIG['tree_and_local_reference_tables']:
                leaves=minimum_tree_leaves(labels,indices);cubes=local_cubes(labels,indices)
                row.update(minimum_tree_leaves=leaves,tree_code_bits=6*leaves-4,
                           **{key:cubes[key] for key in ('median_minimum_literals','median_training_coverage','fraction_sharing_with_another_training_sample','median_full_domain_cube_size')})
                if role=='original':
                    original_details.append({'dataset':dataset,'n':n,'input_ids':indices.tolist(),'labels':labels[indices].tolist(),
                                             'linear':lp,'parity_subset':subset,**cubes})
                    if n in (32,80,128) and LOSS_PAIRING_DIR is not None:
                        for hidden in ('32','32_32','32_32_32'):
                            file=Path(LOSS_PAIRING_DIR)/f'random_d{dataset}_n{n}_h{hidden}'/'observations.npz'
                            with np.load(file,allow_pickle=False) as a:
                                np.testing.assert_array_equal(a['input_ids'],indices)
                                losses=a['node_BCE'][1]
                                profile=(losses/losses.mean(1,keepdims=True)).mean(0)
                            paired.append({'dataset':dataset,'n':n,'hidden':hidden,
                                           'min_literal_vs_loss_spearman':corr(cubes['minimum_literals'],profile),
                                           'local_coverage_vs_loss_spearman':corr(cubes['same_label_training_coverage'],profile)})
            rows.append(row)
        if dataset%16==0:print(role,dataset,'累计',round(time.monotonic()-start,1),'秒',flush=True)
    finished=len({(r['role'],r['dataset']) for r in rows})
    result={'config':CONFIG,'status':'completed' if finished==len(tables) else 'budget_stopped',
            'seconds':time.monotonic()-start,'finished_tables':finished,'rows':rows,'original_details':original_details,'loss_pairing':paired}
    atomic_json(ROOT/'RESULTS.json',result)
    report(result)
    print('完成',result['status'],finished,'份表，',result['seconds'],'秒')


def report(result):
    rates=[]
    for n in COUNTS:
        group=[r for r in result['rows'] if r['role']=='reference' and r['n']==n]
        if not group:continue
        structural=[r for r in group if 'minimum_tree_leaves' in r]
        lp=[r['linear_separable'] for r in group if r['linear_separable'] is not None]
        rates.append({'n':n,'tables':len(group),'constant_or_literal':float(np.mean([r['constant_or_literal_fit'] for r in group])),
                      'parity':float(np.mean([r['parity_fit'] for r in group])),
                      'linear':float(np.mean(lp)) if lp else None,'linear_resolved':len(lp),
                      'tree_tables':len(structural),'median_tree_leaves':float(np.median([r['minimum_tree_leaves'] for r in structural])) if structural else None,
                      'median_local_literals':float(np.median([r['median_minimum_literals'] for r in structural])) if structural else None,
                      'median_local_coverage':float(np.median([r['median_training_coverage'] for r in structural])) if structural else None,
                      'local_reuse_fraction':float(np.mean([r['fraction_sharing_with_another_training_sample'] for r in structural])) if structural else None})
    atomic_json(ROOT/'SUMMARY.json',rates)
    lines=['# 随机训练集的系统性捷径排查','',
           f"状态：{result['status']}，共{result['finished_tables']}份随机表，耗时{result['seconds']:.1f}秒。仅CPU，不运行训练或SMC。",
           '原四份数据保持不变；另外128份独立随机表作参照。每张表按随机排列形成嵌套前缀。常数/单bit、任意affine parity和仿射阈值检查覆盖全部参照表；树与局部子立方体统计用前32份参照表。',
           '', '## 整份训练集能否被简单结构完全拟合','',
           '线性分类器指8个输入加bias的仿射阈值，以LP检查是否存在严格正margin；不表示已优化范数或找到了最短程序。parity允许常数翻转。',
           '', '| n | 随机表数 | 常数或单bit | 任意parity | 线性分类器 | 决策树最少叶数中位数 |','|---|---:|---:|---:|---:|---:|']
    for r in rates:
        linear='未判定' if r['linear'] is None else f"{100*r['linear']:.1f}%"
        tree='未测' if r['median_tree_leaves'] is None else f"{r['median_tree_leaves']:.1f}"
        lines.append(f"| {r['n']} | {r['tables']} | {100*r['constant_or_literal']:.1f}% | {100*r['parity']:.1f}% | {linear} | {tree} |")
    lines += ['', '这些概率针对本次8-bit均匀输入、随机标签和指定结构族，不是所有任务的普适临界n。不同n共享随机表，不能当独立重复；同n的不同表才是独立单位。',
              '', '## 即使整份不能简单拟合，局部是否仍能复用','',
              '对每个训练点，找最少几个bit取值条件，使对应子立方体不包含任何已见异标签点。同样少的bit中选覆盖同标签训练点最多者。',
              '这是明确的局部规则见证，但没有检查它在未见随机标签上是否正确，也没有证明MLP实际采用该分支。',
              '', '| n | 结构参照表数 | 最少条件bit数中位数 | 局部覆盖训练点数中位数 | 可与另一训练点共享该分支的比例 |','|---|---:|---:|---:|---:|']
    for r in rates:
        if r['tree_tables']:
            lines.append(f"| {r['n']} | {r['tree_tables']} | {r['median_local_literals']:.1f} | {r['median_local_coverage']:.1f} | {100*r['local_reuse_fraction']:.1f}% |")
    lines += ['', '## 原四份数据','', '| 表 | n | 常数或单bit | parity | 线性可分 | 最小树叶数 | 局部条件bit | 局部覆盖点 |','|---|---:|---|---|---|---:|---:|---:|']
    for row in result['rows']:
        if row['role']!='original' or row['n'] not in (8,16,32,80,128):continue
        lines.append(f"| {row['dataset']} | {row['n']} | {row['constant_or_literal_fit']} | {row['parity_fit']} | {row['linear_separable']} | {row['minimum_tree_leaves']} | {row['median_minimum_literals']:.1f} | {row['median_training_coverage']:.1f} |")
    lines += ['', '## 与真实训练loss的配对','',
              '下面是三类n、四份原表的探索性相关中位数。局部分支越难描述、或覆盖同类点越少，是否对应较高BCE，需要看符号和稳定性；相关不证明因果。',
              '', '| 隐层 | 最少条件bit数与loss的Spearman中位数 | 局部覆盖量与loss的Spearman中位数 |','|---|---:|---:|']
    for hidden in ('32','32_32','32_32_32'):
        selected=[r for r in result['loss_pairing'] if r['hidden']==hidden]
        cols=[]
        for key in ('min_literal_vs_loss_spearman','local_coverage_vs_loss_spearman'):
            vals=[r[key] for r in selected if r[key] is not None];cols.append(f"{float(np.median(vals)):.3f}" if vals else "未提供配对输入")
        lines.append(f"| {hidden} | {cols[0]} | {cols[1]} |")
    lines += ['', '## 结论边界','',
              '多换随机seed能平均随机波动，不能消除“样本太少时普遍存在简单插值”这一系统性性质。',
              '随机生成机制与有限训练集存在便宜相容结构可以同时成立。这里检测训练集意义上的捷径，不认证外部生成规律。',
              '存在局部分支、或者用少量树叶拟合，不等于总编码一定比n个标签位短；选择结构和例外也有成本。树的固定前缀码示例为每内节点4bit、每叶2bit，L叶共6L-4bit，但它不是最短Boolean程序。',
              '排除这几类简单结构不能证明不可压缩，也不能排除更深网络特有的计算复用。',
              '没有更换SMC数据、没有按结果筛掉任何随机表、没有把这些代理拟合成新的复杂度公式。']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择任务；本模块不自动计算。")
