import unittest
from unittest import TestCase
from scipy.stats import skew, kurtosis
from array import array
from sherlock.features.stats_helper import compute_stats
from numpy.testing import assert_array_almost_equal
import numpy as np
from timeit import timeit


def old_stats(values):
    m = np.mean(values)
    v = np.var(values)
    s = skew(values)
    k = kurtosis(values)

    return m, v, s, k


class Test(TestCase):

    def test_custom_stats(self):
        all_values = [array('i', [1, 1, 1, 1, 1, 1, 1]),
                      array('i', [1, 0, 0, 0, 1]),
                      array('i', [0, 1, 1, 1, 0]),
                      array('i', [2, 1, 1, 1, 2]),
                      array('i', [2, 0, 0, 0, 1]),
                      array('i', [1, 0, 0, 0, 0]),
                      array('i', [1, 0, 0, 0, 1]),
                      array('i', [1, 1, 1, 2, 1]),
                      array('i', [1, 0, 0, 2, 1]),
                      array('i', [0, 1, 1, 0, 0]),
                      array('i', [0, 0, 0, 1, 0]),
                      array('i', [1, 0, 0, 0, 1]),
                      array('i', [0, 1, 1, 0, 0]),
                      array('i', [1, 0, 0, 0, 0]),
                      array('i', [4, 1, 1, 0, 1]),
                      array('i', [0, 1, 1, 1, 1]),
                      array('i', [1, 1, 1, 1, 2]),
                      array('i', [8, 0, 0, 3, 3]),
                      array('i', [1, 0, 0, 1, 0]),
                      array('i', [1, 0, 0, 0, 0]),
                      array('i', [2, 2, 2, 1, 2]),
                      array('i', [1, 0, 0, 0, 1]),
                      array('i', [6, 2, 2, 1, 0]),
                      array('i', [1, 2, 2, 1, 0]),
                      array('i', [1, 0, 0, 2, 0]),
                      array('i', [4, 2, 2, 1, 3]),
                      array('i', [3, 2, 2, 1, 1]),
                      array('i', [4, 4, 4, 2, 1]),
                      array('i', [1, 1, 1, 1, 0]),
                      array('i', [0, 0, 0, 0, 1])]

        for values in all_values:
            _mean, _variance, _skew, _kurtosis = old_stats(values)

            _mean2, _variance2, _skew2, _kurtosis2 = compute_stats(values)

            assert_array_almost_equal(_mean, _mean2, err_msg=f'mismatch for values {values}')
            assert_array_almost_equal(_variance, _variance2, err_msg=f'mismatch for values {values}')
            assert_array_almost_equal(_skew, _skew2, err_msg=f'mismatch for values {values}')
            assert_array_almost_equal(_kurtosis, _kurtosis2, err_msg=f'mismatch for values {values}')

    @unittest.skip("benchmark - run manually")
    def test_custom_stats_benchmark(self):
        # compute_stats is about 10x faster than using scipy
        # t1=0.539309483, t2=5.406991439
        v = array('i', [4, 4, 4, 2, 1])

        t1 = timeit(lambda: compute_stats(v), number=10000)
        t2 = timeit(lambda: old_stats(v), number=10000)

        print(f't1={t1}, t2={t2}')
