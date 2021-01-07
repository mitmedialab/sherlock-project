from unittest import TestCase
from sherlock.features.word_embeddings import initialise_word_embeddings, extract_word_embeddings_features
from collections import OrderedDict
from tests.test_helper import assert_category_keys


class Test(TestCase):
    @classmethod
    def setUpClass(cls):
        initialise_word_embeddings()

    def test_extract_word_embeddings_features(self):
        col_values = ['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat', 'two words', '!£$@$ one']

        features = OrderedDict()

        extract_word_embeddings_features(col_values, features)

        assert_category_keys('word', features)
