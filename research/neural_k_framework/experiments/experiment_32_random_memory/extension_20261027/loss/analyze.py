"""逐样本loss、跨初始化稳定性和探索性邻域诊断；纯CPU。"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from config import WORK_DIR

ROOT=WORK_DIR / "loss"


def scalar(value):return float(value) if np.isfinite(value) else None


def spearman(a,b):
    ar=rankdata(a).astype(float);br=rankdata(b).astype(float)
    ar-=ar.mean();br-=br.mean();den=np.linalg.norm(ar)*np.linalg.norm(br)
    return float(ar@br/den) if den>0 else None


def pair_rank(values):
    ranks=rankdata(values,axis=1).astype(float);ranks-=ranks.mean(1,keepdims=True)
    norms=np.linalg.norm(ranks,axis=1)
    valid=norms>0
    ranks=ranks[valid]/norms[valid,None]
    if len(ranks)<2:return None
    matrix=ranks@ranks.T
    return float(np.mean(matrix[np.triu_indices(len(ranks),1)]))


def summarize_losses(loss):
    mean=loss.mean(1);normalized=loss/mean[:,None]
    n=loss.shape[1];q=np.quantile(loss,[.1,.5,.9],axis=1)
    p90_p10=np.divide(q[2],q[0],out=np.full(len(loss),np.inf),where=q[0]>0)
    max_min=np.divide(loss.max(1),loss.min(1),out=np.full(len(loss),np.inf),where=loss.min(1)>0)
    top_n=max(1,math.ceil(n*.1));quarter=max(1,math.ceil(n*.25))
    top=np.zeros(loss.shape,dtype=np.int32)
    np.put_along_axis(top,np.argsort(loss,axis=1)[:,-quarter:],1,axis=1)
    overlap=top@top.T/quarter
    off=overlap[np.triu_indices(len(loss),1)]
    return {'models':len(loss),'mean_BCE_median':float(np.median(mean)),
            'minimum_BCE_median':float(np.median(loss.min(1))),'maximum_BCE_median':float(np.median(loss.max(1))),
            'max_to_min_median':scalar(np.median(max_min)),'p90_to_p10_median':scalar(np.median(p90_p10)),
            'max_to_mean_median':float(np.median(normalized.max(1))),
            'sample_CV_median':float(np.median(loss.std(1)/mean)),
            'hardest_10percent_loss_share_median':float(np.median(np.sort(loss,axis=1)[:,-top_n:].sum(1)/loss.sum(1))),
            'pairwise_sample_rank_spearman_mean':pair_rank(loss),
            'hardest_quartile_overlap_mean':float(off.mean()) if len(off) else None,
            'hardest_quartile_random_overlap_reference':quarter/n}


def independent_forward(theta,x,layers):
    value=np.broadcast_to(x.astype(np.float64),(len(theta),*x.shape))
    for i,(start,end,stop,fan_in,fan_out) in enumerate(layers):
        weights=theta[:,start:end].astype(np.float64).reshape(len(theta),fan_out,fan_in)
        value=np.einsum('pni,poi->pno',value,weights)+theta[:,None,end:stop].astype(np.float64)
        if i<len(layers)-1:value=np.tanh(value)
    return value[:,:,0]


def local_support(x,y):
    distance=np.not_equal(x[:,None,:],x[None,:,:]).sum(2)
    weights=np.exp2(-distance.astype(float));np.fill_diagonal(weights,0)
    sign=2*y.astype(float)-1
    return ((weights@sign)*sign)/weights.sum(1)


def main():
    plan=json.loads((ROOT/'PLAN.json').read_text(encoding='utf-8'))
    status=json.loads((ROOT/'STATUS.json').read_text(encoding='utf-8'))
    rows=[];drift=[];verified=0;max_error=0.;sample_profiles=[]
    for path in sorted((ROOT/'results').glob('*/RESULT.json')):
        info=json.loads(path.read_text(encoding='utf-8'))
        raw=path.with_name('observations.npz')
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=info['observations_sha256']:raise RuntimeError('结果文件hash不符')
        with np.load(raw,allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
        x=arrays['inputs'];y=arrays['labels'];initial=arrays['initial_BCE']
        features=local_support(x,y)
        initial_stats=summarize_losses(initial)
        initial_re=np.logaddexp(0.,independent_forward(arrays['initial_theta_raw'],x,arrays['layers'])*(1-2*y.astype(float)))
        np.testing.assert_allclose(initial_re,initial,rtol=1e-11,atol=1e-12)
        profiles={}
        for i,epsilon in enumerate(arrays['node_epsilon']):
            selected=arrays['node_steps'][i]>=0
            if not selected.any():continue
            loss=arrays['node_BCE'][i,selected]
            recomputed=np.logaddexp(0.,independent_forward(arrays['node_theta_raw'][i,selected],x,arrays['layers'])*(1-2*y.astype(float)))
            error=float(np.max(abs(recomputed-loss)));max_error=max(max_error,error)
            if error>1e-11:raise RuntimeError('离线逐样本BCE核验失败')
            if loss.mean(1).max()>epsilon+2e-7:raise RuntimeError('节点BCE不符合阈值')
            if epsilon<math.log(2)/info['n'] and np.any(loss>=math.log(2)):raise RuntimeError('低BCE节点有训练分类错误')
            stats=summarize_losses(loss)
            normalized=loss/loss.mean(1,keepdims=True)
            profile=normalized.mean(0);profiles[float(epsilon)]=profile
            init_final=[spearman(a,b) for a,b in zip(initial[selected],loss)]
            available=[v for v in init_final if v is not None]
            row={**{k:info[k] for k in ('key','kind','dataset','n','hidden','status')},'epsilon':float(epsilon),
                 'all_models_reached':bool(selected.all()),**stats,
                 'initial_rank_agreement':initial_stats['pairwise_sample_rank_spearman_mean'],
                 'same_seed_initial_final_rank_correlation_mean':float(np.mean(available)) if available else None,
                 'local_support_vs_loss_rank':spearman(features,profile)}
            rows.append(row);verified+=len(loss)
            if epsilon==.001 and info['dataset']==0 and info['n']==128:
                for index in range(len(y)):
                    sample_profiles.append({'kind':info['kind'],'hidden':info['hidden'],'input_id':int(arrays['input_ids'][index]),
                                            'bits':''.join(map(str,x[index].astype(int))),'label':int(y[index]),
                                            'mean_normalized_BCE':float(profile[index]),'local_support':float(features[index]),
                                            'BCE_by_initialization':loss[:,index].tolist()})
        if .01 in profiles and .001 in profiles and .0001 in profiles:
            drift.append({**{k:info[k] for k in ('key','kind','dataset','n','hidden')},
                          'rank_01_to_001':spearman(profiles[.01],profiles[.001]),
                          'rank_001_to_0001':spearman(profiles[.001],profiles[.0001])})
    metrics=['mean_BCE_median','minimum_BCE_median','maximum_BCE_median','max_to_min_median','p90_to_p10_median',
             'max_to_mean_median','sample_CV_median','hardest_10percent_loss_share_median','pairwise_sample_rank_spearman_mean',
             'hardest_quartile_overlap_mean','initial_rank_agreement','same_seed_initial_final_rank_correlation_mean','local_support_vs_loss_rank']
    grouped=[]
    for kind in ('random','projection0'):
        for hidden in plan['config']['architectures']:
            for n in plan['config']['counts']:
                for epsilon in plan['config']['nodes']:
                    selected=[r for r in rows if r['kind']==kind and r['hidden']==hidden and r['n']==n and r['epsilon']==epsilon and r['all_models_reached']]
                    if not selected:continue
                    aggregate={'kind':kind,'hidden':hidden,'n':n,'epsilon':epsilon,'complete_datasets':len(selected)}
                    for metric in metrics:
                        values=[r[metric] for r in selected if r[metric] is not None]
                        aggregate[metric]=float(np.median(values)) if values else None
                    grouped.append(aggregate)
    result={'status':status,'plan':plan,'case_nodes':rows,'groups':grouped,'rank_drift':drift,'sample_profiles':sample_profiles,
            'audited_node_model_snapshots':verified,'max_independent_BCE_error':max_error,
            'aggregation':'每个模型计算逐样本指标，先16模型中位数，再四数据表中位数；未齐全表不混入正式聚合。',
            'scope':'训练后的平均loss预算分配与易难稳定性；不是最短描述长度、SMC系数或真实生成规律的认证。'}
    (ROOT/'ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 随机记忆样本的逐样本BCE差异','',
           f"状态：{status['status']}；已产生{len(list((ROOT/'results').glob('*/RESULT.json')))}组/45组。",
           '重新生成并逐项验证原四份随机表，n=32、80、128，隐藏层32、32/32、32/32/32，每组16个初始化。Adam lr=.003，无权重衰减，完整批次训练。',
           '正对照为相同输入上的projection0，使用随机表0相同的初始权重。每个模型第一次达到相同平均BCE时分别保存，另存初始化逐样本loss。',
           'GPU1 RTX3080；float32训练，float64逐样本BCE核验；不运行SMC。',
           '', '## 本轮结论', '',
           '随机标签下逐样本BCE有很大差距，而且容易/困难样本的排序跨初始化重复。这不是只有某次初始化偶然挑出了几个困难点。',
           '较难样本占据不成比例的loss预算，且平均BCE进一步降低后，相对差异没有自动消失。两层和三层都可出现此现象。',
           '这支持把“有限随机训练集内部的样本成本不均匀”纳入解释，但不能直接推出标签不是随机源生成，也不能确认这就是SMC系数凹形的原因。',
           '简单规则对照反而有更均匀的样本loss，因此“loss差距越大就越有规律”不适合作为单一判据。局部同标签支持度与loss的负相关属于进一步线索，不是因果证明。',
           '', '## 随机表主窗口 BCE≤0.001','',
           '以下先按初始化取中位数，再对四份表取中位数。Spearman为同一训练集中，不同初始化的样本loss排序相关。',
           '', '| 隐层 | n | 表数 | 最大/平均loss | P90/P10 | 最难10%贡献 | 跨初始化排序相关 | 最难四分位重叠 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in grouped:
        if g['kind']!='random' or g['epsilon']!=.001:continue
        lines.append(f"| {g['hidden']} | {g['n']} | {g['complete_datasets']} | {g['max_to_mean_median']:.2f} | {g['p90_to_p10_median']:.2f} | {100*g['hardest_10percent_loss_share_median']:.1f}% | {g['pairwise_sample_rank_spearman_mean']:.3f} | {g['hardest_quartile_overlap_mean']:.3f} |")
    lines += ['', '四分位重叠以相交数量/集合大小计；随机独立挑选的参考约0.25。16模型形成的120个模型对共享模型，不当作120个独立重复。',
              '', '## 降低loss后差异是否消失','', '| 类型 | 隐层 | n | 平均BCE节点 | 最小loss | 最大loss | P90/P10 | 排序相关 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for g in grouped:
        if g['n']!=128:continue
        lines.append(f"| {g['kind']} | {g['hidden']} | {g['n']} | {g['epsilon']} | {g['minimum_BCE_median']:.3g} | {g['maximum_BCE_median']:.3g} | {g['p90_to_p10_median']:.2f} | {g['pairwise_sample_rank_spearman_mean']:.3f} |")
    lines += ['', '绝对loss随目标降低而减小，与相对差异消失是不同判断；相同平均BCE并不要求各点margin相同。',
              '', '## 初始化与局部结构诊断','',
              '探索性局部同标签支持度采用固定公式：对其他训练样本按2的负Hamming距离加权，计算同标签/异标签的符号一致度。只做事后诊断，不参与优化，也不是预注册的因果检验。',
              '其与样本loss负相关，表示靠近同类训练样本的位置更易拟合；不能据此认定未见随机标签存在生成依赖。',
              '', '| 类型 | 隐层 | n | 初始化排序共识 | 训练后排序共识 | 同seed初始到最终相关 | 局部支持度与loss相关 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for g in grouped:
        if g['epsilon']!=.001:continue
        lines.append(f"| {g['kind']} | {g['hidden']} | {g['n']} | {g['initial_rank_agreement']:.3f} | {g['pairwise_sample_rank_spearman_mean']:.3f} | {g['same_seed_initial_final_rank_correlation_mean']:.3f} | {g['local_support_vs_loss_rank']:.3f} |")
    lines += ['', '## 判读边界','',
              '此实验直接检验平均BCE内部是否存在不均匀拟合，以及这种易难差异是否跨初始化重复。它能为“平均loss利用了容易样本”提供观测，不单独证明数据包含可复用的标签规律。',
              '纯随机标签的一个有限实现也有局部同类簇、相互冲突的近邻、可用的简单插值结构；真实训练的margin差异还包含输入几何和优化路径。它们都可能造成loss不均。',
              '不能从最大/最小loss比例反推出算法互信息，也不能用训练后的loss把测试集规律判为真实。不同深度的差异是现象，尚未成为三层SMC收缩曲线的机制解释。',
              f"全部已达节点共核验{verified}个模型快照，独立float64逐样本BCE最大误差{max_error:.3g}。原始权重、样本ID及各节点loss全部保留。"]
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    concise=[{k:g[k] for k in ('kind','hidden','epsilon','p90_to_p10_median','pairwise_sample_rank_spearman_mean')}
             for g in grouped if g['n']==128 and g['epsilon']==.001]
    print(json.dumps({'status':status,'n128_main':concise,'audited_snapshots':verified,'error':max_error},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择任务；本模块不自动计算。")
