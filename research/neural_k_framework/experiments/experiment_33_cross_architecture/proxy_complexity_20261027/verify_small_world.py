"""独立枚举所有2-input电路状态，与SAT最小性逐项核对。"""

import itertools
import json
from pathlib import Path
import time

from exact_circuits import LANGUAGES, atom_tables, apply_gate, encode, solve


HERE=Path(__file__).resolve().parent


def exhaustive_costs(language):
    spec=LANGUAGES[language]
    initial,_=atom_tables(2,spec["free_input_negation"])
    states={tuple(sorted(set(initial)))}
    costs={}
    for depth in range(5):
        for state in states:
            for value in state:
                costs.setdefault(value,depth)
                costs.setdefault(15^value,depth)
        if len(costs)==16:
            return costs
        following=set()
        for state in states:
            for op in spec["ops"]:
                pairs=((v,v) for v in state) if op=="not" else itertools.combinations_with_replacement(state,2)
                for a,b in pairs:
                    value=apply_gate(op,a,b,15)
                    if value not in state:
                        following.add(tuple(sorted((*state,value))))
        states=following
    raise RuntimeError("2-input穷举未覆盖全部函数")


def main():
    rows=[]
    started=time.perf_counter()
    for language in LANGUAGES:
        costs=exhaustive_costs(language)
        for target,cost in sorted(costs.items()):
            for budget,status in [(cost,"sat_verified")]+([(cost-1,"unsat_verified")] if cost else []):
                cnf,meta=encode(2,[target],budget,language)
                result=solve(cnf,meta,HERE/"test_artifacts"/"exhaustive2"/language/f"f{target:02x}_r{budget}",30)
                if result["status"]!=status:
                    raise RuntimeError(f"独立枚举不符：{language} f={target} r={budget}")
            rows.append({"language":language,"target":target,"min_gates":cost})
    value={"status":"PASS","functions_per_language":16,"languages":list(LANGUAGES),
           "comparisons":len(rows),"seconds":time.perf_counter()-started,"rows":rows}
    (HERE/"SMALL_WORLD_VERIFICATION.json").write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in value.items() if k!="rows"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
