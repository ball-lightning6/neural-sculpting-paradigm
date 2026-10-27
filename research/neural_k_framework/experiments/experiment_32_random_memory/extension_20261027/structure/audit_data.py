"""固定Boolean描述字典和GF(2)检查；不给有限样本贴绝对纯噪声标签。"""

from itertools import combinations, product
import json
import math
from pathlib import Path

import numpy as np
from config import WORK_DIR
from data import random_table

ROOT=WORK_DIR / "structure"
COUNTS=(16,32,48,64,80,96,112,128)


def clog2(value):return (int(value)-1).bit_length()


def dictionary(bits):
    """三位族编码；族内显式编码结构，完整函数重复时保留最短描述。"""
    funcs={}
    def add(pred,cost,description):
        pred=np.asarray(pred,dtype=np.uint8)
        key=np.packbits(pred,bitorder='little').tobytes()
        if key not in funcs or cost<funcs[key][1]:funcs[key]=(pred,cost,description)
    for value in (0,1):add(np.full(256,value),4,f'constant_{value}')
    for k in range(1,9):
        choose=clog2(math.comb(8,k))
        for subset in combinations(range(8),k):
            parity=bits[:,subset].sum(1)%2
            for complement in (0,1):add(parity^complement,3+4+choose+1,f'parity{subset}_flip{complement}')
            for signs in product((0,1),repeat=k):
                literals=bits[:,subset]==np.array(signs)
                code=3+4+choose+k
                add(literals.all(1),code,f'AND{subset}_values{signs}')
                add(literals.any(1),code,f'OR{subset}_values{signs}')
                if k in (3,5,7):add(literals.sum(1)>k//2,code,f'majority{subset}_values{signs}')
    for root,left,right in product(range(8),repeat=3):
        slot=2*bits[:,root]+np.where(bits[:,root]>0,bits[:,right],bits[:,left])
        for mask in range(16):add((mask>>slot)&1,16,f'depth2_tree_{root}_{left}_{right}_leaves{mask}')
    for selector in range(8):
        rest=[j for j in range(8) if j!=selector]
        for mask in range(128):
            subset=[rest[j] for j in range(7) if (mask>>j)&1]
            parity=bits[:,subset].sum(1).astype(np.uint8)%2 if subset else np.zeros(256,np.uint8)
            for gate,constant,flip in product((0,1),repeat=3):
                pred=np.where(bits[:,selector]==gate,constant,parity^flip)
                add(pred,16,f'gate{selector}={gate}_const{constant}_parity{subset}_flip{flip}')
    values=list(funcs.values())
    predictions=np.array([x[0] for x in values],dtype=np.uint8)
    costs=np.array([x[1] for x in values],dtype=np.int64)
    descriptions=[x[2] for x in values]
    kraft=float(np.exp2(-costs.astype(float)).sum())
    if kraft>1+1e-12:raise RuntimeError('描述字典不满足Kraft约束')
    return predictions,costs,descriptions,kraft


def gf2_consistent(matrix,labels):
    basis={}
    for values,label in zip(matrix,labels):
        word=int.from_bytes(np.packbits(values,bitorder='little').tobytes(),'little')
        target=int(label)
        while word:
            pivot=word.bit_length()-1
            if pivot not in basis:
                basis[pivot]=(word,target);break
            row,rhs=basis[pivot];word^=row;target^=rhs
        if not word and target:return False
    return True


def polynomial_features(bits):
    columns=[];degrees=[]
    for degree in range(9):
        for subset in combinations(range(8),degree):
            columns.append(bits[:,subset].all(1) if subset else np.ones(256,bool));degrees.append(degree)
    return np.array(columns,dtype=np.uint8).T,np.array(degrees)


def full_structure(bits,labels):
    index=(bits*(1<<np.arange(8))).sum(1).astype(int)
    truth=np.empty(256,dtype=np.uint8);truth[index]=labels
    coeff=truth.copy()
    for j in range(8):
        for mask in range(256):
            if mask&(1<<j):coeff[mask]^=coeff[mask^(1<<j)]
    terms=np.flatnonzero(coeff)
    return {"ones":int(labels.sum()),"anf_terms":int(len(terms)),
            "anf_degree":max((int(i).bit_count() for i in terms),default=0),
            "total_influence":float(sum(np.mean(truth!=truth[np.arange(256)^(1<<j)]) for j in range(8)))}


def main():
    bits=(random_table(0)[0]>0).astype(np.uint8)
    pred,cost,desc,kraft=dictionary(bits)
    features,degrees=polynomial_features(bits)
    rows=[];structures=[]
    for dataset in range(4):
        inputs,labels,order=random_table(dataset)
        np.testing.assert_array_equal(bits,(inputs>0).astype(np.uint8))
        structures.append({"dataset":dataset,**full_structure(bits,labels)})
        for n in COUNTS:
            selected=order[:n];remaining=order[n:]
            errors=(pred[:,selected]!=labels[selected]).sum(1)
            error_code=np.array([clog2(n+1)+clog2(math.comb(n,int(k))) for k in range(n+1)])
            # 首位区分原始标签编码和模型编码，结构与例外位置都计费。
            total=1+cost+error_code[errors]
            best=int(np.argmin(total))
            degree=next(d for d in range(9) if gf2_consistent(features[selected][:,degrees<=d],labels[selected]))
            rows.append({"dataset":dataset,"n":n,"ones":int(labels[selected].sum()),"shortest_dictionary_code_bits":int(total[best]),
                         "raw_label_code_bits":n+1,"saving_vs_raw_bits":int(n+1-total[best]),"best_rule":desc[best],
                         "rule_cost_bits":int(cost[best]),"errors":int(errors[best]),
                         "unseen_accuracy_after_training_only_selection":float(np.mean(pred[best,remaining]==labels[remaining])),
                         "minimum_interpolating_anf_degree":degree})
    result={"dictionary_size":len(pred),"dictionary_kraft_sum":kraft,"full_tables":structures,"prefixes":rows,
            "scope":"有限字典正例可展示一种压缩；阴性结果不能证明不可压缩，也不能排除三层网络特有的便宜实现。",
            "definition":"1+模型结构码长+ceil(log2(n+1))+ceil(log2(C(n,错误数)))；原始标签1+n。输入及顺序双方已知。"}
    (ROOT/'DATA_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 随机记忆数据结构检查','',f'固定字典共{len(pred)}个不同Boolean函数，Kraft和{kraft:.6f}。',
           '覆盖常数、全部parity、带符号AND/OR、3/5/7输入多数函数、两层判断树、常数分支与parity分支的组合。',
           '选择规则只看训练标签；未见标签只用于事后评分。不是直接拿训练拟合率当压缩。','',
           '| 表 | 完整表中1的个数 | ANF项数 | ANF最高次数 | 总影响度 |','|---|---:|---:|---:|---:|']
    for s in structures:lines.append(f"| {s['dataset']} | {s['ones']} | {s['anf_terms']} | {s['anf_degree']} | {s['total_influence']:.3f} |")
    lines += ['', '| 表 | n | 最短模型加残差码长 | 相对原始标签节省bit | 训练错误 | 未见准确率 | 可插值最低ANF次数 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['dataset']} | {r['n']} | {r['shortest_dictionary_code_bits']} | {r['saving_vs_raw_bits']} | {r['errors']} | {r['unseen_accuracy_after_training_only_selection']:.3f} | {r['minimum_interpolating_anf_degree']} |")
    lines += ['', '## 如何读这些量','',
              '节省bit为负表示该字典没有比逐位存储更省，实际编码会选原始标签分支；不是数据有负复杂度。',
              '最低ANF次数只是存在性代理：低次数可包含大量单项式及自由系数，不能把次数低直接说成描述很短。',
              '有限随机样本可恰好满足简单规则。只有把选择哪条规则的成本计入，才能谈相对于该描述语言的压缩。',
              '上述字典不是神经网络全部可实现函数；未找到压缩不能排除该架构通过共享中间表示实现便宜的训练插值。',
              '结合AGREEMENT.md看训练延拓是否集中，再结合未见标签正确率；三者分开，不据此直接修改K的定义。']
    (ROOT/'DATA_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('字典',len(pred),'Kraft',kraft,'最大节省',max(r['saving_vs_raw_bits'] for r in rows))


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择任务；本模块不自动计算。")
