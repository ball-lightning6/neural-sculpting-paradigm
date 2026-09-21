"""520节点：固定数值策略；不读取SMC参考值，无argparse。"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'neural_k_backtest_v1'
CORE=ROOT.parent/'neural_k_residual_v3'
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(CORE))
from memory_safe_cgf import MemorySafeCGF
from conditional_integral import TiltedHiddenSampler,aggregate

CONFIG={'device':'cuda:1','batch_size':8192,'repeats':4,'defensive':.1,
        'initial_samples':2**14,'refinement_samples':2**16,'maximum_samples':2**18}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def identity(p):
    return p['bits'],p['function_id'],p['epsilon']


def diagnostic_reasons(summary,baseline):
    reasons=[]
    if summary is None or summary.get('K_bits') is None:
        return ['NO_FINITE_INTEGRAL']
    if summary['min_ESS']<1000:
        reasons.append('LOW_ESS')
    if summary['max_weight_share']>.01:
        reasons.append('LARGE_SINGLE_WEIGHT')
    if summary['max_internal_K_standard_error']>.05:
        reasons.append('LARGE_INTERNAL_SE')
    if summary['replica_K_range']>.3:
        reasons.append('REPLICA_DISAGREEMENT')
    if not baseline['quadrature_stable']:
        reasons.append('BASE_QUADRATURE_FLAG')
    if not baseline['numerically_converged']:
        reasons.append('BASE_OPTIMIZER_FLAG')
    return reasons


def finalize(case):
    last=case['stages'][-1]
    summary=last.get('summary')
    reasons=diagnostic_reasons(summary,case['baseline'])
    case.update(conditional_pass=not reasons,diagnostic_flags=reasons,
                conditional_K_bits=summary.get('K_bits') if summary else None,
                estimate_K_bits=summary['K_bits'] if not reasons else case['baseline']['saddle_K_bits'],
                selected_method='conditional_integral' if not reasons else 'saddlepoint_unconfirmed_fallback',
                final_samples_per_replica=last['sample_count'],completed=True)
    return case


def progress(cases,total,elapsed,phase,current='',start_text=''):
    complete=[x for x in cases if x.get('completed')]
    passed=sum(x['conditional_pass'] for x in complete)
    payload={'phase':phase,'completed':len(complete),'total':total,'conditional_pass':passed,
             'fallback':len(complete)-passed,'current':current,'elapsed_seconds_this_process':elapsed,
             'wall_clock_local':time.strftime('%Y-%m-%d %H:%M:%S')}
    write(ROOT/'progress.json',payload)
    text=f'''# 大规模估算回测进度

- 状态：{phase}
- 完成：{len(complete)}/{total}节点
- 条件积分内部筛查通过：{passed}
- 保留快速近似并标记未确认：{len(complete)-passed}
- 当前：{current or '无'}
- 本次进程耗时：{elapsed/60:.1f}分钟
- 更新：{payload['wall_clock_local']}

范围：3bit全部40轨道、4bit预选64轨道，各5档BCE。原测量未作为调参目标。原始条件积分及失败保留。
'''
    (ROOT/'PROGRESS.md').write_text(text,encoding='utf-8')


def load_reusable():
    result={}
    for folder in ('pilot','extended'):
        path=CORE/folder/'predictions_frozen.json'
        manifest=json.loads((path.parent/'manifest.json').read_text())
        if sha(path)!=manifest['prediction_sha256'] or sha(CORE/'conditional_integral.py')!=manifest['source_sha256']:
            raise RuntimeError('旧积分的冻结哈希不符。')
        for row in json.loads(path.read_text()):
            result[identity(row)]={'row':row,'source_path':str(path),'source_sha256':sha(path)}
    return result


def next_sample_count(stage,baseline):
    summary=stage.get('summary')
    n=stage['sample_count']
    if not diagnostic_reasons(summary,baseline) or n>=CONFIG['maximum_samples']:
        return None
    if not baseline['quadrature_stable'] or not baseline['numerically_converged']:
        return None
    if n<CONFIG['refinement_samples']:
        return CONFIG['refinement_samples']
    if summary and summary.get('K_bits') is not None and summary['min_ESS']>=64 and summary['max_weight_share']<=.2 and summary['replica_K_range']<=3:
        return CONFIG['maximum_samples']
    return None


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    sources=[ROOT/'run_benchmark.py',ROOT/'PROTOCOL.md',CORE/'conditional_integral.py',OLD/'finite_cgf.py',OLD/'memory_safe_cgf.py']
    source_hashes={str(p):sha(p) for p in sources}
    validation=json.loads((CORE/'validation.json').read_text())
    if not validation['all_passed'] or validation['source_sha256']!=sha(CORE/'conditional_integral.py'):
        raise RuntimeError('原条件积分核心需通过核验。')
    inputs=[OLD/'three_bit_v1/predictions_frozen.json',OLD/'four_bit_v1/refined_predictions_frozen.json']
    predictions=[]
    for p in inputs:
        predictions.extend(json.loads(p.read_text()))
    if len(predictions)!=520 or len(set(identity(p) for p in predictions))!=520:
        raise RuntimeError('预定节点清单不完整。')
    fingerprint=hashlib.sha256(json.dumps({'sources':source_hashes,'config':CONFIG},sort_keys=True).encode()).hexdigest()
    selection={'source_hashes':source_hashes,'input_hashes':{str(p):sha(p) for p in inputs},'fingerprint':fingerprint,
               'config':CONFIG,'reference_read_by_predictor':False,
               'cases':[{k:p[k] for k in ('bits','function_id','epsilon','order')} for p in predictions]}
    if (ROOT/'selection_frozen.json').exists():
        prior=json.loads((ROOT/'selection_frozen.json').read_text())
        if prior!=selection:
            raise RuntimeError('配置或源码发生变化，不能混合旧结果。')
    else:
        write(ROOT/'selection_frozen.json',selection)
    write(ROOT/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'device':CONFIG['device'],'gpu':torch.cuda.get_device_name(1),'dtype':'float64','tf32':False,
        'memory_total':torch.cuda.get_device_properties(1).total_memory})
    case_dir=ROOT/'cases'
    case_dir.mkdir(exist_ok=True)
    reusable=load_reusable()
    results={}
    for path in case_dir.glob('*.json'):
        row=json.loads(path.read_text())
        if row['fingerprint']!=fingerprint:
            raise RuntimeError('节点来源与本轮不同。')
        results[identity(row)]=row
    begin=time.perf_counter()
    model=None
    model_key=None
    for p in sorted(predictions,key=lambda x:(x['bits'],x['order'],x['function_id'],x['epsilon'])):
        key=identity(p)
        name=f"b{p['bits']}_f{p['function_id']:04x}_e{p['epsilon']}"
        path=case_dir/(name+'.json')
        if key in results and results[key].get('completed'):
            continue
        baseline={k:p[k] for k in ('saddle_K_bits','action_bits','quadrature_stable','quadrature_change_bits','numerically_converged','seconds')}
        case=results.get(key,{**{k:p[k] for k in ('bits','function_id','epsilon','order')},'baseline':baseline,'fingerprint':fingerprint,'stages':[]})
        results[key]=case
        if not case['stages'] and key in reusable:
            old=reusable[key]
            records=old['row']['records']
            case['stages'].append({'name':'reused_v3','sample_count':records[0]['sample_count'],'records':records,
                'summary':aggregate(records),'reused_source':old['source_path'],'reused_sha256':old['source_sha256']})
        stage=case['stages'][-1] if case['stages'] else None
        if stage and len(stage['records'])==4:
            n=next_sample_count(stage,baseline)
        elif stage:
            n=stage['sample_count']
        else:
            n=CONFIG['initial_samples']
        sampler=None
        try:
            while n is not None:
                if model_key!=(p['bits'],p['order']):
                    if model is not None:
                        del model
                        torch.cuda.empty_cache()
                    model_key=(p['bits'],p['order'])
                    model=MemorySafeCGF(bits=p['bits'],order=p['order'],device=CONFIG['device'])
                if sampler is None:
                    sampler=TiltedHiddenSampler(model,p)
                if not stage or len(stage['records'])==4:
                    stage={'name':f'new_N{n}','sample_count':n,'records':[],'summary':None}
                    case['stages'].append(stage)
                for repeat in range(len(stage['records']),CONFIG['repeats']):
                    progress(list(results.values()),520,time.perf_counter()-begin,'计算中',f'{name}, N={n}, 副本{repeat+1}')
                    seed=int.from_bytes(hashlib.sha256(f'20260915:{key}:{n}:{repeat}'.encode()).digest()[:8],'little')%(2**63-1)
                    record=sampler.replica(n,seed,CONFIG['batch_size'],CONFIG['defensive'])
                    stage['records'].append(record)
                    if len(stage['records'])==4:
                        stage['summary']=aggregate(stage['records'])
                    write(path,case)
                n=next_sample_count(stage,baseline)
        except (ArithmeticError,RuntimeError) as exc:
            case['numerical_exception']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
            if not case['stages']:
                case['stages'].append({'name':'exception','sample_count':0,'records':[],'summary':None})
            elif len(case['stages'][-1]['records'])<4:
                case['stages'][-1]['summary']=None
        finally:
            if sampler is not None:
                del sampler
        finalize(case)
        if case.get('numerical_exception'):
            case.update(conditional_pass=False,selected_method='saddlepoint_unconfirmed_fallback',estimate_K_bits=p['saddle_K_bits'])
            case['diagnostic_flags'].append('NUMERICAL_EXCEPTION')
        write(path,case)
        complete=sum(x.get('completed',False) for x in results.values())
        if complete%10==0 or case.get('numerical_exception'):
            passed=sum(x.get('conditional_pass',False) for x in results.values())
            print(f'{complete}/520 通过={passed} {name} N={case["final_samples_per_replica"]} flags={case["diagnostic_flags"]} 本次{(time.perf_counter()-begin)/60:.1f}分钟',flush=True)
    final=[results[identity(p)] for p in sorted(predictions,key=identity)]
    write(ROOT/'predictions_frozen.json',final)
    write(ROOT/'manifest.json',{'fingerprint':fingerprint,'prediction_sha256':sha(ROOT/'predictions_frozen.json'),
        'case_count':len(final),'conditional_pass':sum(r['conditional_pass'] for r in final),
        'elapsed_seconds_this_process':time.perf_counter()-begin,'frozen_time_local':time.strftime('%Y-%m-%d %H:%M:%S')})
    progress(final,520,time.perf_counter()-begin,'预测已冻结，等待测量对比')
    print('520节点冻结完成。计算阶段没有读取SMC。',flush=True)


if __name__=='__main__':
    main()
