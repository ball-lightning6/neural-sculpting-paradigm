"""穷举有限函数空间，核验软候选竞争推导；不运行神经网络实验。"""

import itertools
import math
import unittest


def compatible(function, target, inputs):
    return all(((function ^ target) >> x) & 1 == 0 for x in inputs)


def survival(n, disagreements, domain_size, replacement):
    if replacement:
        return (1.0 - disagreements / domain_size) ** n
    if n > domain_size - disagreements:
        return 0.0
    return math.comb(domain_size - disagreements, n) / math.comb(domain_size, n)


class CandidateCompetitionTests(unittest.TestCase):
    def test_addition_deletion_costs_are_conditional_surprise(self):
        # 有限系综恒等式检查，不是神经网络的规则/噪声实验。
        domain_size = 3
        weights = [float((f + 1) ** 2) for f in range(1 << domain_size)]
        prior = [w / sum(weights) for w in weights]
        for target in range(len(prior)):
            def mass(inputs):
                return sum(p for f, p in enumerate(prior) if compatible(f, target, inputs))

            for n in range(domain_size):
                for inputs in itertools.combinations(range(domain_size), n):
                    before = mass(inputs)
                    for point in set(range(domain_size)) - set(inputs):
                        after = mass((*inputs, point))
                        addition = -math.log2(after) + math.log2(before)
                        deletion = math.log2(before / after)
                        self.assertAlmostEqual(addition, -math.log2(after / before))
                        self.assertAlmostEqual(deletion, addition)
                        self.assertGreaterEqual(addition, -1e-12)
            for order in itertools.permutations(range(domain_size)):
                total = sum(-math.log2(mass(order[:i + 1]) / mass(order[:i]))
                            for i in range(domain_size))
                self.assertAlmostEqual(total, -math.log2(prior[target]))

    def test_equal_accuracy_does_not_require_equal_error_locations(self):
        # 三个函数各错一个不同测试点，准确率完全相同但预测不相同。
        functions = (0b001, 0b010, 0b100)
        accuracy = [1 - f.bit_count() / 3 for f in functions]
        self.assertEqual(len(set(accuracy)), 1)
        self.assertEqual(len(set(functions)), 3)
        pointwise_agreement = (1 / 3) ** 2 + (2 / 3) ** 2
        self.assertAlmostEqual(pointwise_agreement, 5 / 9)
        self.assertAlmostEqual(sum(accuracy) / 3, 2 / 3)

    def test_survival_jensen_and_markov(self):
        domain_size = 4
        weights = [(f + 1) ** 2 for f in range(1 << domain_size)]
        prior = [weight / sum(weights) for weight in weights]
        nonvacuous_bounds = 0
        for replacement in (False, True):
            for n in range(6 if replacement else domain_size + 1):
                draw = itertools.product if replacement else itertools.combinations
                datasets = list(
                    draw(range(domain_size), repeat=n)
                    if replacement
                    else draw(range(domain_size), n)
                )
                for target, target_mass in enumerate(prior):
                    rivals, posterior = [], []
                    for inputs in datasets:
                        mass = sum(
                            prior[f]
                            for f in range(len(prior))
                            if f != target and compatible(f, target, inputs)
                        )
                        rivals.append(mass)
                        posterior.append(target_mass / (target_mass + mass))
                    expected = sum(
                        prior[f]
                        * survival(n, (f ^ target).bit_count(), domain_size, replacement)
                        for f in range(len(prior))
                        if f != target
                    )
                    with self.subTest(replacement=replacement, n=n, target=target):
                        self.assertAlmostEqual(sum(rivals) / len(rivals), expected)
                        proxy = target_mass / (target_mass + expected)
                        self.assertLessEqual(proxy, sum(posterior) / len(posterior) + 1e-12)
                        for threshold in (0.1, 0.25, 0.5, 0.75, 0.9):
                            failure = sum(q < threshold for q in posterior) / len(posterior)
                            bound = min(1.0, threshold / (1 - threshold) * expected / target_mass)
                            self.assertLessEqual(failure, bound + 1e-12)
                            nonvacuous_bounds += 0 < bound < 1
        self.assertGreater(nonvacuous_bounds, 0)

    def test_without_replacement_near_full_domain(self):
        self.assertAlmostEqual(survival(240, 1, 256, False), 0.0625)
        self.assertGreater(survival(240, 1, 256, True), 0.39)

    def test_single_flip_neighbor_bound(self):
        domain_size = 4
        masses = [(f + 1) ** 2 for f in range(1 << domain_size)]
        for n in range(domain_size):
            for inputs in itertools.combinations(range(domain_size), n):
                unseen = set(range(domain_size)) - set(inputs)
                for target, target_mass in enumerate(masses):
                    total = sum(p for f, p in enumerate(masses) if compatible(f, target, inputs))
                    ratio = min(masses[target ^ (1 << x)] / target_mass for x in unseen)
                    bound = 1 / (1 + len(unseen) * ratio)
                    self.assertLessEqual(target_mass / total, bound + 1e-12)

    def test_mass_sum_is_not_minimum_selection(self):
        masses = [1.0] + [1.0 / 32] * 1000
        total = sum(masses)
        prior = [mass / total for mass in masses]
        costs = [-math.log2(mass) for mass in prior]
        minimum = min(costs)
        soft_cost = minimum - math.log2(sum(2 ** (minimum - cost) for cost in costs))
        self.assertAlmostEqual(soft_cost, -math.log2(sum(prior)))
        self.assertAlmostEqual(prior[0], 1 / 32.25)
        self.assertLess(prior[0], 0.032)

    def test_target_mass_monotonicity_does_not_imply_agreement_monotonicity(self):
        # 函数 ID 的第 x 位就是输入 x 的标签；新观察是 x=0, y=1。
        prior = [0.9, 0.05, 0.0, 0.05]
        target = 3
        normalizer = sum(p for f, p in enumerate(prior) if compatible(f, target, (0,)))
        posterior = [p / normalizer if compatible(f, target, (0,)) else 0 for f, p in enumerate(prior)]

        def mean_agreement(distribution):
            total = 0.0
            for x in range(2):
                p_one = sum(p for f, p in enumerate(distribution) if (f >> x) & 1)
                total += p_one ** 2 + (1 - p_one) ** 2
            return total / 2

        self.assertGreater(posterior[target], prior[target])
        self.assertLess(mean_agreement(posterior), mean_agreement(prior))
        self.assertLess(sum(p ** 2 for p in posterior), sum(p ** 2 for p in prior))


if __name__ == "__main__":
    unittest.main(verbosity=2)
