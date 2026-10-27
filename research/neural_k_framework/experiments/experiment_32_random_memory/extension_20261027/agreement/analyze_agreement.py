"""从逐模型预测独立核验agreement，并用固定探针和后续标签检查复用性。"""

import json
from pathlib import Path

import numpy as np
from config import WORK_DIR

ROOT=WORK_DIR / "agreement"
COUNTS=(16,32,48,64,80,96,112,128)


def metrics(pred,labels):
    count=len(pred)
    votes=pred.sum(0,dtype=np.int64)
    pairs=(votes*(votes-1)+(count-votes)*(count-votes-1))/(count*(count-1))
    majority=np.where(votes>count/2,1.,np.where(votes<count/2,0.,.5))
    majority_correct=np.where(majority==.5,.5,majority==labels)
    _,counts=np.unique(pred,axis=0,return_counts=True)
    return {"agreement":float(pairs.mean()),"model_accuracy":float((pred==labels).mean()),
            "vote_accuracy":float(majority_correct.mean()),"predicted_one_fraction":float(pred.mean()),
            "probe_function_collision":float(np.sum(counts*(counts-1))/(count*(count-1))),
            "probe_function_modal_fraction":float(counts.max()/count),"distinct_probe_functions":len(counts)}


def main():
    source=json.loads((ROOT/'AGREEMENT.json').read_text(encoding='utf-8'))
    rows=[];errors=[]
    for result in source['rows']:
        info=result['identity'];file=ROOT/'agreement'/info['key']/'predictions.npz'
        with np.load(file,allow_pickle=False) as a:
            pred=a['predictions'];labels=a['labels'];order=a['order'];steps=a['step_at_hit'];loss=a['loss_at_hit']
        hit=steps[-1]>=0
        if not hit.all():continue
        p=pred[-1].astype(bool)
        if not (p[:,order[:info['n']]]==labels[order[:info['n']]]).all():raise RuntimeError('主节点未硬拟合')
        if np.max(loss[-1])>.001:raise RuntimeError('实际BCE超阈值')
        check=metrics(p[:,order[info['n']:]],labels[order[info['n']:]])
        error=abs(check['agreement']-result['nodes'][-1]['pair_agreement']);errors.append(error)
        if error>1e-12:raise RuntimeError('agreement独立核验不一致')
        # 固定末128点；避免n增大时探针集合自己变化造成趋势。
        common=order[128:]
        row={"key":info['key'],"kind":info['kind'],"dataset":info['dataset'],"n":info['n'],"hidden":info['hidden'],
             "steps":result['steps'],"seconds":result['seconds'],"models":len(p),"common_probe":metrics(p[:,common],labels[common]),
             "full_256_function":metrics(p,labels)}
        if info['n']<128:
            subsequent=order[info['n']:info['n']+16]
            votes=p[:,subsequent].sum(0)
            probability=(votes+.5)/(len(p)+1.)
            truth_probability=np.where(labels[subsequent],probability,1-probability)
            row['next_16']={**metrics(p[:,subsequent],labels[subsequent]),
                            "log_loss_bits":float(-np.log2(truth_probability).sum()),
                            "probability_smoothing":"(正票数+0.5)/(模型数+1)，固定，不用测试标签校准"}
        rows.append(row)
    groups=[];sequential=[]
    for kind in ('random','projection0'):
        for hidden in ((32,),(32,32),(32,32,32)):
            for n in COUNTS:
                selected=[r for r in rows if r['kind']==kind and r['hidden']==list(hidden) and r['n']==n]
                if not selected:continue
                groups.append({"kind":kind,"hidden":list(hidden),"n":n,"datasets":len(selected),
                               **{key:float(np.mean([r['common_probe'][key] for r in selected]))
                                  for key in ('agreement','model_accuracy','vote_accuracy','predicted_one_fraction','probe_function_collision','probe_function_modal_fraction')}})
            for dataset in range(4 if kind=='random' else 1):
                selected=sorted((r for r in rows if r['kind']==kind and r['hidden']==list(hidden) and r['dataset']==dataset),key=lambda r:r['n'])
                if [r['n'] for r in selected]!=list(COUNTS):continue
                total=16+sum(r['next_16']['log_loss_bits'] for r in selected[:-1])
                sequential.append({"kind":kind,"hidden":list(hidden),"dataset":dataset,"cumulative_log_loss_bits_including_first16":total,
                                   "saving_relative_to_128_fair_coin_bits":128-total})
    output={"source_status":source['status'],"elapsed_seconds":source['elapsed_seconds'],"verified_groups":len(rows),
            "max_agreement_recompute_error":max(errors,default=0.),"rows":rows,"means":groups,"sequential_prediction":sequential,
            "scope":"顺序预测分数衡量前缀规律对下一块标签是否有用；不测同训练集上的内部表示共享，更不测参数体积。"}
    (ROOT/'AGREEMENT_ANALYSIS.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 训练agreement与可复用结构','',f"已完成并独立核验{len(rows)}组，每组32个初始化；训练累计{source['elapsed_seconds']/60:.2f}分钟。",
           '主节点BCE<=.001，均保留实际第一次穿越值。下面全部使用固定末128个未见输入，不随训练样本数改变探针集合。',
           '逐点agreement与整张探针真值表的碰撞率分开报告；完整256点的函数碰撞及模态占比也保存在JSON中。',
           '', '| 类型 | 隐层 | n | 表数 | 逐点agreement | 探针函数碰撞率 | 模型平均正确率 | 多数投票正确率 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['kind']} | {g['hidden']} | {g['n']} | {g['datasets']} | {g['agreement']:.4f} | {g['probe_function_collision']:.4f} | {g['model_accuracy']:.4f} | {g['vote_accuracy']:.4f} |")
    lines += ['', '## 前缀对后续标签的顺序预测','',
              '先将前16个标签逐位计16bit；之后用n=16,32,...112的已训练模型预测紧接着的16个标签。概率为固定平滑后的投票频率。',
              '总对数损失小于128bit，说明该固定预测器在这份实现上获得了顺序预测收益；大于128bit说明共识并未换来相对公平硬币的预测收益。',
              '这是预先固定模型的评分，不是求最短程序；不把后验选择最好的架构或随机种子的收益当免费。',
              '', '| 类型 | 隐层 | 表 | 总分bit | 相对128bit节省 |','|---|---|---:|---:|---:|']
    for s in sequential:lines.append(f"| {s['kind']} | {s['hidden']} | {s['dataset']} | {s['cumulative_log_loss_bits_including_first16']:.2f} | {s['saving_relative_to_128_fair_coin_bits']:.2f} |")
    lines += ['', '## 能回答和不能回答的事','',
              '能检验当前随机表在三层网络下是否突然出现更集中的延拓，及这种延拓是否对后续标签有用。',
              '即使未见预测没有提高，也不能排除三层网络在已经给定的训练约束内部共享计算、扩大低loss实现区域；那是另一种“便宜”。',
              '四份表与一种固定Adam协议只能提供配对筛查。此处不以一个绝对agreement阈值给出“存在规律”的置信概率。',
              '不同模型对共享同一模型，不能当作496个独立重复；当前报告以32个初始化和四份数据表为独立层级。',
              '没有重现n=1/2的早期世界实验；这里n从16开始，不据此改写旧实验结论。']
    (ROOT/'AGREEMENT_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('已核验',len(rows),'组；耗时分钟',source['elapsed_seconds']/60)
    for s in sequential:print(s)


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择任务；本模块不自动计算。")
