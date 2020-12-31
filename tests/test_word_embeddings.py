from unittest import TestCase
from sherlock.features.word_embeddings import initialise_word_embeddings, extract_word_embeddings_features
import pandas as pd
from collections import OrderedDict
from tests.test_helper import assert_category_keys


class Test(TestCase):
    @classmethod
    def setUpClass(cls):
        initialise_word_embeddings()

    def test_extract_word_embeddings_features(self):
        s = pd.Series(['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat'])

        features = OrderedDict()

        extract_word_embeddings_features(s, features)

        assert_category_keys('word', features)
