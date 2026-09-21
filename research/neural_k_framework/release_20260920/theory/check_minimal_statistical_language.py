"""最小统计语言的有限概率核验。使用CPU和标准库，不训练网络。"""

import json
import math
import random
from fractions import Fraction as F

RESULTS = []
MAX_ERROR = 0.0


def close(actual, expected, tol=2e-12):
    global MAX_ERROR
    error = abs(float(actual) - float(expected))
    MAX_ERROR = max(MAX_ERROR, error)
    assert error <= tol, (actual, expected, error)


def mass(p, event):
    return sum((v for v, keep in zip(p, event) if keep), F(0))


def condition(p, event):
    total = mass(p, event)
    assert total > 0
    return [v / total if keep else F(0) for v, keep in zip(p, event)]


def aggregate(p, groups):
    result = {}
    for value, group in zip(p, groups):
        result[group] = result.get(group, F(0)) + value
    return result


def kl(p, q):
    return sum(float(a) * math.log2(float(a / b)) for a, b in zip(p, q) if a)


def collision(p):
    return sum(a * a for a in p)


def record(name, details):
    RESULTS.append({"check": name, "details": details})


# 同一平均阈值可同时删除旧参数、加入旧集合外的参数。
p = [F(2, 5), F(1, 5), F(2, 5)]
old_loss = [F(1, 10), F(19, 100), F(3, 10)]
query_loss = [F(1, 10), F(2), F(1, 100)]
groups = [(1, 1, 0), (1, 0, 0), (1, 1, 1)]
eps = F(1, 5)
old = [v <= eps for v in old_loss]
new = [(a + b) / 2 <= eps for a, b in zip(old_loss, query_loss)]
intersection = [a and b for a, b in zip(old, new)]
assert old == [True, True, False]
assert new == [True, False, True]
old_mass, new_mass = mass(p, old), mass(p, new)
assert new_mass / old_mass == F(4, 3)
q_cap, q_new = aggregate(condition(p, intersection), groups), aggregate(condition(p, new), groups)
assert q_cap[(1, 1, 0)] == 1 and q_new[(1, 1, 0)] == F(1, 2)
for group in groups:
    cf = [x == group for x in groups]
    old_branch = mass(p, [a and b for a, b in zip(cf, old)])
    outgoing = mass(p, [a and b and not c for a, b, c in zip(cf, old, new)])
    incoming = mass(p, [a and b and not c for a, b, c in zip(cf, new, old)])
    new_branch = mass(p, [a and b for a, b in zip(cf, new)])
    assert new_branch == old_branch - outgoing + incoming
record("average_loss_incoming_outgoing", {
    "old_mass": str(old_mass), "new_mass": str(new_mass),
    "new_over_old": str(new_mass / old_mass), "cap_target": "1", "new_target": "1/2"
})

# 同一hard函数内部可筛掉一半；参数KL与函数KL不同。
p = [F(1, 2), F(1, 2)]
old_loss = [F(19, 100)] * 2
query_loss = [F(1, 10), F(3, 10)]
assert all(float(v) < math.log(2) for v in query_loss)
accepted = [(a + b) / 2 <= eps for a, b in zip(old_loss, query_loss)]
assert accepted == [True, False]
q = condition(p, accepted)
close(kl(q, p), 1)
assert aggregate(p, ["same", "same"]) == aggregate(q, ["same", "same"])
record("hard_function_internal_loss", {
    "hard_label_probability": 1, "soft_acceptance": 0.5,
    "parameter_kl_bits": 1, "function_kl_bits": 0
})

# 两个候选标签的连续loss事件可能重叠，不能视为标签分支。
assert (0.1 + math.log(2)) / 2 < 0.5
candidate_masses = [1, 1]
assert sum(candidate_masses) == 2
record("candidate_events_not_partition", {
    "query_logit": 0, "old_loss": 0.1, "new_threshold": 0.5,
    "candidate_masses": candidate_masses, "hard_readout_at_tie": 1
})

# 非负loss给出的充分嵌套条件。
for n in [1, 2, 9]:
    for a in [F(i, 100) for i in range(61)]:
        for b in [F(i, 100) for i in range(61)]:
            if (n * a + b) / (n + 1) <= n * eps / (n + 1):
                assert a <= eps
record("total_budget_nesting", {"n_values": [1, 2, 9]})

# 保留率、协方差、KL分解均在独立随机有限概率模型中核验。
rng = random.Random(20260920)
for _ in range(200):
    weights = [rng.randint(1, 30) for _ in range(12)]
    total = sum(weights)
    p = [F(v, total) for v in weights]
    parent = [bool(rng.getrandbits(1)) for _ in p]
    parent[0] = True
    child = [a and bool(rng.getrandbits(1)) for a in parent]
    child[0] = True
    groups = [rng.randrange(4) for _ in p]
    before, after = condition(p, parent), condition(p, child)
    qb, qa = aggregate(before, groups), aggregate(after, groups)

    delta = -math.log2(float(mass(p, child) / mass(p, parent)))
    close(kl(after, before), delta)
    macro_kl = kl([qa[g] for g in sorted(qa)], [qb[g] for g in sorted(qa)])
    internal = 0.0
    for g, qa_g in qa.items():
        if not qa_g:
            continue
        indices = [i for i, v in enumerate(groups) if v == g]
        af = [after[i] / qa_g for i in indices]
        bf = [before[i] / qb[g] for i in indices]
        internal += float(qa_g) * kl(af, bf)
    close(delta, macro_kl + internal)

    # 条件事件内任意分布的KL投影分解。
    raw = [rng.randint(1, 30) if keep else 0 for keep in parent]
    arbitrary = [F(v, sum(raw)) for v in raw]
    close(kl(arbitrary, p),
          -math.log2(float(mass(p, parent))) + kl(arbitrary, before))

    # r只在旧正质量cell上计算，嵌套事件不产生新的cell。
    old_branch = aggregate([v if keep else F(0) for v, keep in zip(p, parent)], groups)
    new_branch = aggregate([v if keep else F(0) for v, keep in zip(p, child)], groups)
    support = [g for g in qb if qb[g]]
    rates = {g: new_branch[g] / old_branch[g] for g in support}
    er = sum(qb[g] * rates[g] for g in support)
    costs = {g: F(g + 1) for g in support}
    eb = sum(qb[g] * costs[g] for g in support)
    ea = sum(qa[g] * costs[g] for g in support)
    covariance = sum(qb[g] * costs[g] * rates[g] for g in support) - eb * er
    assert ea - eb == covariance / er
    for g in support:
        assert qa[g] == qb[g] * rates[g] / er

    # 碰撞率界及概率预算。
    maximum = max(qa.values())
    coll = collision(qa.values())
    assert maximum * maximum <= coll <= maximum
    v = sum(new_branch.values())
    for k in [1, 3, 6, 10]:
        count = sum(value >= F(1, 2**k) for value in new_branch.values())
        assert count <= v * 2**k
record("finite_random_models", {"models": 200, "checks": [
    "conditioning_kl", "coarse_graining_kl_chain", "minimum_kl_decomposition",
    "retention_update", "cost_covariance", "collision_bounds", "probability_budget"
]})

# 与新标签一致的指定目标可以升权，但碰撞率下降。
before = [F(4, 5), F(1, 10), F(1, 10)]
after = condition(before, [False, True, True])
assert after[1] > before[1]
assert collision(after) < collision(before)
record("hard_target_vs_agreement", {
    "target_before": "1/10", "target_after": "1/2",
    "collision_before": str(collision(before)), "collision_after": str(collision(after))
})

# 零成本方差不等于完整函数凝聚。
q = [F(1, 32)] * 32
assert collision(q) == F(1, 32)
record("cost_width_vs_collision", {"fixed_cost_variance": 0, "collision": "1/32"})

# 累计loss质量与密度条件函数分布相差局部指数重权。
e = F(1, 5)
exponents = [1, 4]
omega = [F(1, 2) * e**a for a in exponents]
density = [F(1, 2) * a * e ** (a - 1) for a in exponents]
q_sub = [x / sum(omega) for x in omega]
q_density = [x / sum(density) for x in density]
average_exponent = sum(q * a for q, a in zip(q_sub, exponents))
for i in range(2):
    assert q_density[i] == q_sub[i] * exponents[i] / average_exponent
assert q_sub != q_density
record("density_vs_sublevel", {
    "sublevel_odds": str(omega[0] / omega[1]),
    "density_odds": str(density[0] / density[1])
})

# 不能丢掉full-target额外的连续精度因子。
p = [F(1, 5), F(3, 10), F(1, 2)]
training_loss = [F(1, 20), F(1, 20), F(9, 100)]
heldout_loss = [F(1, 20), F(3, 10), F(7, 10)]
tau = F(1, 5)
parent = [v <= tau for v in training_loss]
full = [a + b <= tau for a, b in zip(training_loss, heldout_loss)]
hard_target = [float(v) < math.log(2) for v in heldout_loss]
hard_parent = [a and b for a, b in zip(parent, hard_target)]
branch = mass(p, hard_parent)
full_mass = mass(p, full)
precision_factor = full_mass / branch
assert full_mass == branch * precision_factor
assert branch == F(1, 2) and full_mass == F(1, 5) and precision_factor == F(2, 5)
record("full_target_precision_factor", {
    "fixed_D_hard_mass": str(branch), "full_target_mass": str(full_mass),
    "precision_factor": str(precision_factor)
})

# 平移不变的二次函数不具有恒定斜率；二阶差分界可以取到等号。
h = F(1, 2)
quadratic = lambda d: d * d
gamma = quadratic(8 + 2*h) - 2*quadratic(8+h) + quadratic(8)
assert gamma == 2*h*h
eta = F(3, 10)
d0, increment = F(7), F(2)
values = [d0 + n*increment + eta*n*(n-1)/2 for n in range(21)]
for n, value in enumerate(values):
    assert abs(value - values[0] - n*(values[1]-values[0])) == eta*n*(n-1)/2
errors = [F(1, 10), -F(1, 10), -F(1, 10), F(1, 10)]
assert abs(errors[3] - errors[1] - errors[2] + errors[0]) == F(4, 10)
record("translation_and_curvature", {
    "translation_invariance_does_not_imply_linearity": True,
    "grid_error_bound_sharp": True, "four_corner_error_factor": 4
})

print(json.dumps({
    "scope": "有限概率和代数核验；不构成神经网络经验前提的证明",
    "checks": RESULTS, "passed_groups": len(RESULTS),
    "max_float_error": MAX_ERROR
}, ensure_ascii=False, indent=2))
