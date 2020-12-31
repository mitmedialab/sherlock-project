import unittest
from sherlock.functional import normalise_float
import pandas as pd


class FunctionalTestCase(unittest.TestCase):
    def test_normalise_float_remove_imprecision(self):
        result = normalise_float(1.35)

        self.assertEqual('1.35', result)

    def test_normalise_float_scientific(self):
        result = normalise_float(0.0000001)

        self.assertEqual('1e-07', result)

    def test_handle_none(self):
        series = pd.Series(['$', None, 'Aab$', '$$ca', 'C$B$', 'cat', '-0.12'])

        series = series.apply(lambda s: '' if s is None else str(s))

        print(len(series.iloc[1]))
        print(series.iloc[1] == 'None')


if __name__ == '__main__':
    unittest.main()
