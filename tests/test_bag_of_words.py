from unittest import TestCase
from sherlock.features.bag_of_words import extract_bag_of_words_features
import pandas as pd
from collections import OrderedDict
from tests.test_helper import assert_category_keys


class Test(TestCase):
    def test_extract_bag_of_words_features(self):
        s = pd.Series(['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat'])

        features = OrderedDict()

        extract_bag_of_words_features(s, features, len(s))

        assert_category_keys('rest', features)
