"""检查预定选择策略、预算路径、随机流和复用来源。"""

import copy
import hashlib
import json
from pathlib import Path

from run_benchmark import CONFIG,OLD,ROOT,diagnostic_reasons,finalize,identity,load_reusable,next_sample_count,sha,write


def main():
    checks=[]
    def ck(name,ok):
        checks.append({'name':name,'passed':bool(ok)})
    b={'quadrature_stable':True,'numerically_converged':True,'saddle_K_bits':10.}
    s={'K_bits':11.,'min_ESS':2000.,'max_weight_share':.001,'max_internal_K_standard_error':.02,'replica_K_range':.1}
    ck('good_passes',not diagnostic_reasons(s,b))
    for field,value,flag in [('min_ESS',10,'LOW_ESS'),('max_weight_share',.2,'LARGE_SINGLE_WEIGHT'),
                            ('max_internal_K_standard_error',.5,'LARGE_INTERNAL_SE'),('replica_K_range',2.,'REPLICA_DISAGREEMENT')]:
        t={**s,field:value}
        ck(flag,flag in diagnostic_reasons(t,b))
    ck('quadrature_flag','BASE_QUADRATURE_FLAG' in diagnostic_reasons(s,{**b,'quadrature_stable':False}))
    ck('missing_fails',diagnostic_reasons(None,b)==['NO_FINITE_INTEGRAL'])
    case={'baseline':b,'stages':[{'sample_count':16384,'summary':s}]}
    ck('pass_selects_conditional',finalize(copy.deepcopy(case))['estimate_K_bits']==11.)
    case['stages'][0]['summary']={**s,'min_ESS':1}
    ck('failure_falls_back',finalize(copy.deepcopy(case))['estimate_K_bits']==10.)
    ck('pass_no_extra_budget',next_sample_count({'sample_count':16384,'summary':s},b) is None)
    ck('failed_initial_refines',next_sample_count({'sample_count':16384,'summary':{**s,'min_ESS':1}},b)==65536)
    ck('degenerate_refined_stops',next_sample_count({'sample_count':65536,'summary':{**s,'min_ESS':1}},b) is None)
    ck('moderate_refined_continues',next_sample_count({'sample_count':65536,'summary':{**s,'min_ESS':500}},b)==262144)
    ck('maximum_budget_stops',next_sample_count({'sample_count':262144,'summary':{**s,'min_ESS':1}},b) is None)
    predictions=[]
    for name in ('three_bit_v1/predictions_frozen.json','four_bit_v1/refined_predictions_frozen.json'):
        predictions.extend(json.loads((OLD/name).read_text()))
    ck('520_unique',len(predictions)==len({identity(p) for p in predictions})==520)
    seeds=[]
    for p in predictions:
        for n in (16384,65536,262144):
            for repeat in range(4):
                seeds.append(int.from_bytes(hashlib.sha256(f'20260915:{identity(p)}:{n}:{repeat}'.encode()).digest()[:8],'little')%(2**63-1))
    ck('6240_distinct_random_streams',len(set(seeds))==len(seeds)==6240)
    reused=load_reusable()
    ck('21_frozen_reuse_cases',len(reused)==21)
    ck('reused_extended_preserved',reused[(3,105,.001)]['row']['records'][0]['sample_count']==262144)
    result={'all_passed':all(c['passed'] for c in checks),'checks':checks,'runner_sha256':sha(ROOT/'run_benchmark.py')}
    write(ROOT/'validation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['all_passed']:
        raise AssertionError('调度核验失败。')


if __name__=='__main__':
    main()
