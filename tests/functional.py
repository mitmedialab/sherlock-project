import unittest
from sherlock.functional import normalise_float

class FunctionalTestCase(unittest.TestCase):
    def test_normalise_float_remove_imprecision(self):
        result = normalise_float(1.35)

        self.assertEqual('1.35', result)

    def test_normalise_float_scientific(self):
        result = normalise_float(0.0000001)

        self.assertEqual('1e-07', result)


if __name__ == '__main__':
    unittest.main()
