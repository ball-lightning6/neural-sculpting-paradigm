"""用小规模直接枚举核验三进制动态规划与边界。"""

import itertools
import math
import unittest

import numpy as np

from minimax.analyze import decode_dataset, observed_counts, partial_minima


class TestEnumeration(unittest.TestCase):
    def test_against_brute_force(self):
        rng = np.random.default_rng(20261026)
        for m in (1, 2, 3, 4, 5):
            cost = rng.normal(size=2 ** m)
            actual = partial_minima(cost)
            functions = np.arange(2 ** m)
            for index in range(3 ** m):
                mask, labels, observations = decode_dataset(index, m)
                expected = np.min(cost[(functions & mask) == labels])
                self.assertEqual(actual[index], expected)
                self.assertEqual(len(observations), observed_counts(m)[index])

    def test_counts_and_extremes(self):
        m = 8
        costs = np.arange(2 ** m, dtype=np.float64)
        counts = observed_counts(m)
        table = partial_minima(costs)
        maxima = []
        for n in range(m + 1):
            self.assertEqual(np.count_nonzero(counts == n), math.comb(m, n) * 2 ** n)
            maxima.append(np.max(table[counts == n]))
        self.assertEqual(maxima[0], np.min(costs))
        self.assertEqual(maxima[-1], np.max(costs))
        self.assertTrue(np.all(np.diff(maxima) >= 0))

    def test_missing_bounds_and_monotonicity_in_cost(self):
        low = np.array([1., 3., 2., 4.])
        high = np.array([1., np.inf, 2., 7.])
        self.assertTrue(np.all(partial_minima(low) <= partial_minima(high)))
        self.assertEqual(partial_minima(high)[-1], 1.)

    def test_function_label_bit_order(self):
        m = 4
        costs = np.arange(16, dtype=np.float64)
        table = partial_minima(costs)
        for labels in itertools.product((0, 1), repeat=m):
            index = sum(value * 3 ** j for j, value in enumerate(labels))
            function = sum(value * 2 ** j for j, value in enumerate(labels))
            self.assertEqual(table[index], costs[function])


if __name__ == "__main__":
    print("请从 extension_20261027/run.py 显式选择任务；本模块不自动计算。")
