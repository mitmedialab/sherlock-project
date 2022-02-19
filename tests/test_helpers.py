from unittest import TestCase
from sherlock.features.helpers import literal_eval_as_str, keys_to_csv


class Test(TestCase):
    def test_literal_eval_as_str(self):
        s1 = "['Krista Construction Ltd.', None, None, None, \"L'Expert de Parcs J. Aubin Inc.\", 'Lari, Construction', 0.89]"

        result = literal_eval_as_str(s1)

        assert result == ['Krista Construction Ltd.', None, None, None, "L'Expert de Parcs J. Aubin Inc.",
                          'Lari, Construction', '0.89']

    def test_literal_eval_as_str_blank(self):
        s1 = ""

        result = literal_eval_as_str(s1)

        assert result == []

    def test_literal_eval_as_str_empty_array(self):
        s1 = "[]"

        result = literal_eval_as_str(s1)

        assert result == []

    def test_literal_eval_as_str_multiple_commas_in_string(self):
        s1 = "['I have, multiple commas, in string which should, be preserved, ', ', another']"

        result = literal_eval_as_str(s1)

        assert result == ['I have, multiple commas, in string which should, be preserved, ',
                          ', another']

    def test_keys_to_csv(self):
        result = keys_to_csv(['n_[0]-agg-any', 'n_[,]-agg-any', 'n_["]-agg-any'])

        assert result == '"n_[0]-agg-any","n_[,]-agg-any","n_[""]-agg-any"\r\n'
