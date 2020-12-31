from unittest import TestCase
from sherlock.features.bag_of_characters import extract_bag_of_characters_features
import pandas as pd
from collections import OrderedDict
from tests.test_helper import assert_category_keys


class Test(TestCase):
    def test_extract_bag_of_characters_features(self):
        s = pd.Series(['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat'])

        features = OrderedDict()

        extract_bag_of_characters_features(s, features)

        assert_category_keys('char', features)
