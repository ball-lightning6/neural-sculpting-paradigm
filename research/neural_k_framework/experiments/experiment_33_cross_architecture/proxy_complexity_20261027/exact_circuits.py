"""有限Boolean电路的SAT合成和独立正提示LRAT检查，无神经网络依赖。"""

from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


LANGUAGES = {
    "aon_paid": {"ops": ("and", "or", "not"), "free_input_negation": False},
    "aonx_paid": {"ops": ("and", "or", "not", "xor"), "free_input_negation": False},
    "aon_free_inputs": {"ops": ("and", "or", "not"), "free_input_negation": True},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atom_tables(n, free_input_negation=False):
    mask = (1 << (1 << n)) - 1
    values = [0, mask]
    names = ["0", "1"]
    for bit in range(n):
        value = sum(((row >> (n-bit-1)) & 1) << row for row in range(1 << n))
        values.append(value)
        names.append(f"x{bit+1}")
        if free_input_negation:
            values.append(mask ^ value)
            names.append(f"not_x{bit+1}")
    return values, names


def apply_gate(op, a, b, mask):
    if op == "and":
        return a & b
    if op == "or":
        return a | b
    if op == "xor":
        return a ^ b
    if op == "not":
        return mask ^ a
    raise ValueError(op)


def neg(literal):
    return not literal if type(literal) is bool else -literal


class CNF:
    def __init__(self):
        self.variables = 0
        self.clauses = []

    def var(self):
        self.variables += 1
        return self.variables

    def add(self, *literals):
        if any(type(x) is bool and x for x in literals):
            return
        c = tuple(dict.fromkeys(x for x in literals if type(x) is not bool))
        if any(-x in c for x in c):
            return
        self.clauses.append(c)

    def exactly_one(self, xs):
        self.add(*xs)
        # 顺序计数器：s_i表示至少一个前缀选项已取真。
        if len(xs) < 2:
            return
        state = [self.var() for _ in xs[:-1]]
        self.add(-xs[0], state[0])
        for i in range(1, len(xs)-1):
            self.add(-xs[i], state[i])
            self.add(-state[i-1], state[i])
            self.add(-xs[i], -state[i-1])
        self.add(-xs[-1], -state[-1])

    def write(self, path):
        with Path(path).open("w", encoding="ascii", newline="\n") as f:
            f.write(f"p cnf {self.variables} {len(self.clauses)}\n")
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")


def gate_clauses(cnf, selector, output, op, a, b):
    if op == "and":
        clauses = [(neg(a), neg(b), output), (a, -output), (b, -output)]
    elif op == "or":
        clauses = [(a, b, -output), (neg(a), output), (neg(b), output)]
    elif op == "xor":
        clauses = [(a,b,-output), (neg(a),neg(b),-output), (a,neg(b),output), (neg(a),b,output)]
    elif op == "not":
        clauses = [(a,output), (neg(a),-output)]
    else:
        raise ValueError(op)
    for clause in clauses:
        cnf.add(-selector, *clause)


def encode(n, targets, budget, language):
    spec = LANGUAGES[language]
    full_mask = (1 << (1 << n)) - 1
    if any(not 0 <= target <= full_mask for target in targets):
        raise ValueError("目标超出真值表范围")
    atoms, atom_names = atom_tables(n, spec["free_input_negation"])
    cnf = CNF()
    nodes, selections = [], []
    for gate in range(budget):
        size = len(atoms) + gate
        choices = []
        for op in spec["ops"]:
            pairs = ((a,a) for a in range(size)) if op == "not" else itertools.combinations_with_replacement(range(size), 2)
            for a,b in pairs:
                choices.append({"op":op, "a":a, "b":b, "selector":cnf.var()})
        output = [cnf.var() for _ in range(1 << n)]
        cnf.exactly_one([c["selector"] for c in choices])
        for row, out in enumerate(output):
            def literal(index):
                return bool((atoms[index] >> row) & 1) if index < len(atoms) else nodes[index-len(atoms)][row]
            for choice in choices:
                gate_clauses(cnf, choice["selector"], out, choice["op"], literal(choice["a"]), literal(choice["b"]))
        nodes.append(output)
        selections.append(choices)
    outputs = []
    for target in targets:
        choices = []
        for index in range(len(atoms)+budget):
            for invert in (False, True):
                selector = cnf.var()
                choices.append({"signal":index,"invert":invert,"selector":selector})
                for row in range(1 << n):
                    literal = bool((atoms[index] >> row) & 1) if index < len(atoms) else nodes[index-len(atoms)][row]
                    desired = bool((target >> row) & 1) ^ invert
                    cnf.add(-selector, literal if desired else neg(literal))
        cnf.exactly_one([c["selector"] for c in choices])
        outputs.append(choices)
    metadata = {"n":n,"targets":list(targets),"budget":budget,"language":language,
                "atoms":atoms,"atom_names":atom_names,"gate_choices":selections,"output_choices":outputs}
    return cnf, metadata


def check_assignment(cnf, true_literals):
    assigned = set(true_literals)
    if any(-x in assigned for x in assigned):
        return False
    return all(any(literal in assigned for literal in clause) for clause in cnf.clauses)


def decode_witness(metadata, assigned):
    mask = (1 << (1 << metadata["n"])) - 1
    values = metadata["atoms"].copy()
    gates = []
    for choices in metadata["gate_choices"]:
        chosen = [c for c in choices if c["selector"] in assigned]
        if len(chosen) != 1:
            raise RuntimeError("非唯一门选择")
        c = chosen[0]
        value = apply_gate(c["op"], values[c["a"]], values[c["b"]], mask)
        gates.append({k:c[k] for k in ("op","a","b")})
        values.append(value)
    outputs = []
    for choices, target in zip(metadata["output_choices"], metadata["targets"]):
        chosen = [c for c in choices if c["selector"] in assigned]
        if len(chosen) != 1:
            raise RuntimeError("非唯一输出选择")
        c = chosen[0]
        value = values[c["signal"]] ^ (mask if c["invert"] else 0)
        if value != target:
            raise RuntimeError("SAT见证未实现完整目标")
        outputs.append({"signal":c["signal"],"invert":c["invert"]})
    return {"n":metadata["n"],"language":metadata["language"],"atoms":metadata["atom_names"],
            "gates":gates,"outputs":outputs,"targets":metadata["targets"]}


def read_dimacs(path):
    clauses, current, expected = [], [], None
    with Path(path).open(encoding="ascii") as f:
        for line in f:
            if line.startswith("c") or not line.strip():
                continue
            if line.startswith("p"):
                _,kind,variables,count = line.split()
                if kind != "cnf":
                    raise ValueError("非CNF")
                expected = (int(variables),int(count))
                continue
            for literal in map(int,line.split()):
                if literal:
                    current.append(literal)
                else:
                    clauses.append(tuple(current))
                    current = []
    if expected is None or current or len(clauses) != expected[1]:
        raise ValueError("DIMACS计数错误")
    if any(abs(lit)>expected[0] for c in clauses for lit in c):
        raise ValueError("DIMACS变量越界")
    return clauses


def rup_check(clause, hints, database):
    assignment = {}
    for literal in clause:
        variable, value = abs(literal), literal < 0
        if variable in assignment and assignment[variable] != value:
            return True
        assignment[variable] = value
    for hint in hints:
        if hint <= 0:
            raise ValueError("不支持RAT负提示，拒绝认证")
        if hint not in database:
            raise ValueError("提示引用不存在的子句")
        pending = []
        satisfied = False
        for literal in database[hint]:
            value = assignment.get(abs(literal))
            if value is None:
                pending.append(literal)
            elif value == (literal > 0):
                satisfied = True
                break
        if satisfied:
            continue
        pending = list(dict.fromkeys(pending))
        if not pending:
            return True
        if len(pending) != 1:
            raise ValueError("提示不是单元子句")
        assignment[abs(pending[0])] = pending[0] > 0
    return False


def check_lrat(cnf_path, proof_path):
    database = {i+1:clause for i,clause in enumerate(read_dimacs(cnf_path))}
    additions, deletions, empty = 0, 0, False
    max_id = len(database)
    with Path(proof_path).open(encoding="ascii") as f:
        for line_no,line in enumerate(f,1):
            parts = line.split()
            if not parts or parts[0] == "c":
                continue
            idx = int(parts[0])
            if len(parts)>1 and parts[1] == "d":
                if parts[-1] != "0":
                    raise ValueError("删除行缺少终止符")
                for old in map(int,parts[2:-1]):
                    database.pop(old,None)
                    deletions += 1
                continue
            values = list(map(int,parts[1:]))
            split = values.index(0)
            clause = tuple(values[:split])
            hints = values[split+1:]
            if not hints or hints[-1] != 0 or 0 in hints[:-1]:
                raise ValueError("LRAT提示格式错误")
            if idx <= max_id:
                raise ValueError("LRAT新增ID没有递增")
            if not rup_check(clause,hints[:-1],database):
                raise ValueError(f"第{line_no}行RUP推导失败")
            database[idx] = clause
            max_id = idx
            additions += 1
            empty |= len(clause) == 0
    if not empty:
        raise ValueError("证明没有推出空子句")
    return {"status":"verified_rup_lrat","additions":additions,"deletions":deletions,
            "cnf_sha256":digest(cnf_path),"proof_sha256":digest(proof_path)}


def solve(cnf, metadata, folder, timeout=30):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cnf_path, proof_path = folder/"problem.cnf", folder/"proof.lrat"
    cnf.write(cnf_path)
    (folder/"encoding.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    solver = shutil.which("cadical")
    if solver is None:
        raise RuntimeError("未找到本机CaDiCaL，不自动安装或换求解器")
    command = [solver,"--plain","--no-binary","--lrat",str(cnf_path.resolve()),str(proof_path.resolve())]
    started = time.perf_counter()
    try:
        response = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except subprocess.TimeoutExpired as error:
        result = {"status":"unknown_timeout","seconds":time.perf_counter()-started,"command":command}
        (folder/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        return result
    elapsed = time.perf_counter()-started
    (folder/"solver.log").write_text(response.stdout+response.stderr,encoding="utf-8")
    result = {"seconds":elapsed,"command":command,"returncode":response.returncode,
              "variables":cnf.variables,"clauses":len(cnf.clauses),"cnf_sha256":digest(cnf_path)}
    if response.returncode == 10:
        assignment = [int(v) for line in response.stdout.splitlines() if line.startswith("v ") for v in line.split()[1:] if v != "0"]
        if not check_assignment(cnf,assignment):
            raise RuntimeError("求解器见证不满足CNF")
        result.update(status="sat_verified",witness=decode_witness(metadata,set(assignment)))
    elif response.returncode == 20:
        result.update(status="unsat_verified",proof_check=check_lrat(cnf_path,proof_path))
    else:
        result["status"] = "unknown_solver_exit"
    (folder/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


def replay_witness(witness):
    spec = LANGUAGES[witness["language"]]
    values,names = atom_tables(witness["n"],spec["free_input_negation"])
    if names != witness["atoms"]:
        raise ValueError("原子约定不符")
    mask = (1 << (1 << witness["n"])) - 1
    for gate in witness["gates"]:
        if gate["op"] not in spec["ops"] or not 0 <= gate["a"] < len(values) or not 0 <= gate["b"] < len(values):
            raise ValueError("电路见证非法")
        values.append(apply_gate(gate["op"],values[gate["a"]],values[gate["b"]],mask))
    outputs = [values[o["signal"]] ^ (mask if o["invert"] else 0) for o in witness["outputs"]]
    if outputs != witness["targets"]:
        raise ValueError("电路输出不符")
    return outputs
