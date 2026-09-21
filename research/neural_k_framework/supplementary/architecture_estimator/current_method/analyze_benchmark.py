"""520节点冻结后读取SMC；比较精度、顺序、收缩系数和诊断覆盖。"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'neural_k_backtest_v1'
CORE=ROOT.parent/'neural_k_residual_v3'
sys.path.insert(0,str(CORE))
from analyze import references,raw_replicas


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def identity(row):
    return row['bits'],row['function_id'],row['epsilon']


def summarize_replicas(records):
    k=np.array([r['K_bits'] for r in records])
    return {'n':len(k),'K_values':k.tolist(),'sd':float(k.std(ddof=1)) if len(k)>1 else None,
            'min':float(k.min()),'max':float(k.max()),'median':float(np.median(k)),
            'mean_volume_K':float(-(logsumexp(-math.log(2)*k)-math.log(len(k)))/math.log(2))}


def metrics(rows,method,target='SMC_K_bits',range_prefix='SMC_'):
    valid=[r for r in rows if r.get(method) is not None and math.isfinite(r[method])]
    if not valid:
        return {'n':0,'requested_n':len(rows)}
    y=np.array([r[target] for r in valid])
    p=np.array([r[method] for r in valid])
    error=p-y
    r2=1-float(error@error/((y-y.mean())@(y-y.mean()))) if len(y)>1 and np.ptp(y)>0 else None
    rho=float(spearmanr(y,p).statistic) if len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0 else None
    interval_rows=[r for r in valid if r.get(range_prefix+'min') is not None]
    pairs=0
    correct=0
    for i,a in enumerate(interval_rows):
        for b in interval_rows[:i]:
            if a[range_prefix+'max']<b[range_prefix+'min']:
                pairs+=1
                correct+=a[method]<b[method]
            elif b[range_prefix+'max']<a[range_prefix+'min']:
                pairs+=1
                correct+=b[method]<a[method]
    return {'n':len(y),'requested_n':len(rows),'mae_bits':float(np.abs(error).mean()),
            'median_abs_difference_bits':float(np.median(np.abs(error))),
            'median_relative_difference':float(np.median(np.abs(error)/np.maximum(np.abs(y),1e-12))),
            'max_abs_difference_bits':float(np.abs(error).max()),'mean_signed_difference_bits':float(error.mean()),
            'R2':r2,'Spearman':rho,
            'inside_observed_SMC_range':sum(r[range_prefix+'min']<=r[method]<=r[range_prefix+'max'] for r in interval_rows),
            'range_available_n':len(interval_rows),'nonoverlapping_reference_pairs':pairs,
            'correct_order_nonoverlapping_pairs':int(correct),'nonoverlapping_pair_accuracy':correct/pairs if pairs else None}


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    pred_path=ROOT/'predictions_frozen.json'
    if sha(pred_path)!=manifest['prediction_sha256']:
        raise RuntimeError('新预测哈希不符。')
    selection=json.loads((ROOT/'selection_frozen.json').read_text())
    for path,digest in selection['source_hashes'].items():
        if sha(Path(path))!=digest:
            raise RuntimeError('预测源码在冻结后改变。')
    cases=json.loads(pred_path.read_text())
    if len(cases)!=520:
        raise RuntimeError('必须先完成预定520节点。')
    refs=references()
    keys={identity(r) for r in cases}
    raw=raw_replicas(keys)
    ew={}
    with (ROOT.parent/'neural_k_precision_v2/comparison.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            ew[(int(r['bits']),int(r['function_id']),float(r['epsilon']))]=float(r['edgeworth_region']) if r['edgeworth_region'] else None
    rows=[]
    for c in cases:
        key=identity(c)
        ref=refs[key]
        stage=c['stages'][-1]
        summary=stage.get('summary') or {}
        smc=summarize_replicas(raw[key+('first',)])
        if abs(smc['median']-ref['K_bits'])>1e-7:
            raise RuntimeError('原参考中位数口径对不上。')
        warnings=ast.literal_eval(ref['warnings']) if ref.get('warnings') else None
        row={'bits':key[0],'function_id':key[1],'function_hex':f'0x{key[1]:04X}','epsilon':key[2],
             'saddlepoint':c['baseline']['saddle_K_bits'],'conditional_raw':c['conditional_K_bits'],
             'guarded_estimate':c['estimate_K_bits'],'edgeworth_existing':ew.get(key),
             'conditional_pass':c['conditional_pass'],'selected_method':c['selected_method'],
             'diagnostic_flags':c['diagnostic_flags'],'sample_count':c['final_samples_per_replica'],
             'conditional_min_ESS':summary.get('min_ESS'),'conditional_max_weight_share':summary.get('max_weight_share'),
             'conditional_max_K_SE':summary.get('max_internal_K_standard_error'),'conditional_replica_range':summary.get('replica_K_range'),
             'CGF_quadrature_ok':c['baseline']['quadrature_stable'],'CGF_quadrature_change':c['baseline']['quadrature_change_bits'],
             'SMC_K_bits':ref['K_bits'],'SMC_mean_volume_K_bits':smc['mean_volume_K'],
             'SMC_min':smc['min'],'SMC_max':smc['max'],'SMC_sd':smc['sd'],'SMC_n':smc['n'],
             'SMC_K_values':smc['K_values'],'SMC_original_warnings':warnings,
             'SMC_no_alarm':ref.get('no_alarm')=='True' if key[0]==4 else None,
             'SMC_recheck_K_bits':ref.get('recheck_K_bits'),
             'new_compute_seconds':sum(r.get('seconds',0.) for s in c['stages'] if s['name']!='reused_v3' for r in s['records']),
             'reused_prior_seconds':sum(r.get('seconds',0.) for s in c['stages'] if s['name']=='reused_v3' for r in s['records'])}
        if key+('recheck',) in raw:
            re=summarize_replicas(raw[key+('recheck',)])
            row.update(SMC_recheck_min=re['min'],SMC_recheck_max=re['max'],SMC_recheck_sd=re['sd'],SMC_recheck_n=re['n'])
        row['guarded_abs_difference']=abs(row['guarded_estimate']-row['SMC_K_bits'])
        row['saddle_abs_difference']=abs(row['saddlepoint']-row['SMC_K_bits'])
        rows.append(row)
    summaries=[]
    methods=('saddlepoint','conditional_raw','guarded_estimate','edgeworth_existing')
    for bits in (3,4):
        for eps in (.5,.2,.08,.01,.001):
            block=[r for r in rows if r['bits']==bits and r['epsilon']==eps]
            scopes={'all':block,'conditional_pass':[r for r in block if r['conditional_pass']]}
            if bits==4:
                scopes['SMC_no_alarm']=[r for r in block if r['SMC_no_alarm']]
                scopes['conditional_pass_and_SMC_no_alarm']=[r for r in block if r['conditional_pass'] and r['SMC_no_alarm']]
            for scope,subset in scopes.items():
                for method in methods:
                    for target in ('SMC_K_bits','SMC_mean_volume_K_bits'):
                        summaries.append({'bits':bits,'epsilon':eps,'scope':scope,'method':method,'target':target,
                                          'conditional_pass_n':sum(r['conditional_pass'] for r in subset),**metrics(subset,method,target)})
    lookup={identity(r):r for r in rows}
    slopes=[]
    for bits in (3,4):
        fids=sorted({r['function_id'] for r in rows if r['bits']==bits})
        for hi,lo in ((.01,.001),(.08,.01)):
            scale=math.log2(hi/lo)
            for fid in fids:
                a,b=lookup[(bits,fid,hi)],lookup[(bits,fid,lo)]
                original_a={r['replica']:r['K_bits'] for r in raw[(bits,fid,hi,'first')]}
                original_b={r['replica']:r['K_bits'] for r in raw[(bits,fid,lo,'first')]}
                paired=np.array([(original_b[k]-original_a[k])/scale for k in original_a.keys() & original_b.keys()])
                out={'bits':bits,'function_id':fid,'function_hex':f'0x{fid:04X}','epsilon_high':hi,'epsilon_low':lo,
                     'SMC_K_bits':(b['SMC_K_bits']-a['SMC_K_bits'])/scale,
                     'SMC_mean_volume_K_bits':(b['SMC_mean_volume_K_bits']-a['SMC_mean_volume_K_bits'])/scale,
                     'SMC_min':float(paired.min()),'SMC_max':float(paired.max()),'SMC_sd':float(paired.std(ddof=1)),
                     'conditional_pass':a['conditional_pass'] and b['conditional_pass'],
                     'SMC_no_alarm':a['SMC_no_alarm'] and b['SMC_no_alarm'] if bits==4 else None}
                for method in methods:
                    out[method]=(b[method]-a[method])/scale if a.get(method) is not None and b.get(method) is not None else None
                slopes.append(out)
    slope_summary=[]
    for bits in (3,4):
        for hi,lo in ((.01,.001),(.08,.01)):
            block=[r for r in slopes if r['bits']==bits and r['epsilon_high']==hi and r['epsilon_low']==lo]
            for scope in ('all','conditional_pass','SMC_no_alarm'):
                if scope=='SMC_no_alarm' and bits==3:
                    continue
                local=[r for r in block if scope=='all' or r[scope]]
                for method in methods:
                    slope_summary.append({'bits':bits,'epsilon_high':hi,'epsilon_low':lo,'scope':scope,'method':method,
                                           **metrics(local,method)})
    coverage=[]
    for bits in (3,4):
        for eps in (.5,.2,.08,.01,.001):
            local=[r for r in rows if r['bits']==bits and r['epsilon']==eps]
            coverage.append({'bits':bits,'epsilon':eps,'n':len(local),'conditional_pass':sum(r['conditional_pass'] for r in local),
                             'numerical_exceptions':sum('NUMERICAL_EXCEPTION' in r['diagnostic_flags'] for r in local),
                             'median_new_seconds':float(np.median([r['new_compute_seconds'] for r in local]))})
    payload={'scope':'3bit全40轨道与4bit预选64轨道，各5loss。不是4bit全1992。',
             'summaries':summaries,'slope_summaries':slope_summary,'coverage':coverage,
             'reference_hashes':{name:sha(OLD/name) for name in ('three_bit_comparison.csv','four_bit_comparison.csv')},
             'notes':['条件积分未通过时，guarded_estimate使用未经本轮精度确认的鞍点回退值。',
                      'SMC副本范围是观测散布，不是系统误差保证。','raw条件积分的全量指标包含不可靠估计，只用于诊断。',
                      '已有Edgeworth覆盖范围小于全表，按n注明；不进入选择策略。']}
    write(ROOT/'comparison.json',{'rows':rows,'slopes':slopes})
    write(ROOT/'summary.json',payload)
    write(ROOT/'raw_SMC_replicas.json',[{'bits':k[0],'function_id':k[1],'epsilon':k[2],'phase':k[3],'records':v} for k,v in raw.items()])
    csv_write(ROOT/'comparison.csv',rows)
    csv_write(ROOT/'slope_comparison.csv',slopes)
    print('COVERAGE',json.dumps(coverage,ensure_ascii=False))
    print('MAIN')
    for s in summaries:
        if s['scope']=='all' and s['method'] in ('saddlepoint','guarded_estimate') and s['target']=='SMC_K_bits':
            print(s['bits'],s['epsilon'],s['method'],'MAE',s.get('mae_bits'),'R2',s.get('R2'),'rho',s.get('Spearman'))
    print('DEEP SLOPES')
    for s in slope_summary:
        if s['scope']=='all' and s['epsilon_low']==.001 and s['method'] in ('saddlepoint','guarded_estimate'):
            print(s)


if __name__=='__main__':
    main()
