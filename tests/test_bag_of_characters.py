from unittest import TestCase
from sherlock.features.bag_of_characters import extract_bag_of_characters_features
from collections import OrderedDict
from tests.test_helper import assert_category_keys


class Test(TestCase):
    def test_extract_bag_of_characters_features(self):
        col_values = ['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat']

        features = OrderedDict()

        extract_bag_of_characters_features(col_values, features)

        assert_category_keys('char', features)

    def test_values_sum(self):
        col_values = ['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat1', 'caret ^', 'backslash \\']

        features = OrderedDict()

        extract_bag_of_characters_features(col_values, features)

        for f, n in [['n_[$]-agg-sum', 6],
                     ['n_[B]-agg-sum', 2],
                     ['n_[a]-agg-sum', 6],
                     ['n_[b]-agg-sum', 2],
                     ['n_[c]-agg-sum', 4],
                     ['n_[C]-agg-sum', 1],
                     ['n_[B]-agg-sum', 2],
                     ['n_[1]-agg-sum', 1],
                     ['n_[r]-agg-sum', 1],
                     ['n_[e]-agg-sum', 1],
                     ['n_[t]-agg-sum', 2],
                     ['n_[k]-agg-sum', 1],
                     ['n_[l]-agg-sum', 1],
                     ['n_[s]-agg-sum', 2],
                     ['n_[h]-agg-sum', 1],
                     ['n_[\\]-agg-sum', 1],
                     ['n_[Z]-agg-sum', 0]]:
            assert f in features, f'Failed to find key "{f}"'
            assert n == features[f], f'Unexpected value for key "{f}"'
