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
