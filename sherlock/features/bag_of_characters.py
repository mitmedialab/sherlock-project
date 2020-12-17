import string
import numpy  as np
import pandas as pd
from scipy.stats import skew, kurtosis
from collections import OrderedDict

zero_series = pd.Series([0], name='Empty')

characters_to_check = (
        ['[' + c + ']' for c in string.printable if c not in ('\n', '\\', '\v', '\r', '\t', '^')]
        + ['[\\\\]', '[\\^]']
)

ZERO_FLOAT64 = np.float64(0)
DEFAULT_KURTOSIS_FLOAT64 = np.float64(-3.0)


# Input: a single column in the form of pandas series
# Output: ordered dictionary holding bag of character features
def extract_bag_of_characters_features(data_no_null):
    all_value_features = OrderedDict()

    # Create a set of unique chars from the string vectors to quickly test whether to perform expensive
    # counting for any given char in the subsequent loop
    char_set = set(''.join(data_no_null.str.join('')))

    for c in characters_to_check:
        if c[1] in char_set:
            all_value_features['n_{}'.format(c)] = data_no_null.str.count(c)
        else:
            all_value_features['n_{}'.format(c)] = zero_series

    f = OrderedDict()

    for value_feature_name, value_features in all_value_features.items():
        if value_features.name == 'Empty':
            has_any = False
        else:
            has_any = any(value_features)

        if has_any:
            f['{}-agg-any'.format(value_feature_name)] = has_any
            f['{}-agg-all'.format(value_feature_name)] = all(value_features)
            f['{}-agg-mean'.format(value_feature_name)] = np.float64(np.mean(value_features))
            f['{}-agg-var'.format(value_feature_name)] = np.float64(np.var(value_features))
            f['{}-agg-min'.format(value_feature_name)] = np.float64(np.min(value_features))
            f['{}-agg-max'.format(value_feature_name)] = np.float64(np.max(value_features))
            f['{}-agg-median'.format(value_feature_name)] = np.float64(np.median(value_features))
            f['{}-agg-sum'.format(value_feature_name)] = np.float64(np.sum(value_features))
            f['{}-agg-kurtosis'.format(value_feature_name)] = np.float64(kurtosis(value_features))
            f['{}-agg-skewness'.format(value_feature_name)] = np.float64(skew(value_features))
        else:
            f['{}-agg-any'.format(value_feature_name)] = False
            f['{}-agg-all'.format(value_feature_name)] = False
            f['{}-agg-mean'.format(value_feature_name)] = ZERO_FLOAT64
            f['{}-agg-var'.format(value_feature_name)] = ZERO_FLOAT64
            f['{}-agg-min'.format(value_feature_name)] = ZERO_FLOAT64
            f['{}-agg-max'.format(value_feature_name)] = ZERO_FLOAT64
            f['{}-agg-median'.format(value_feature_name)] = ZERO_FLOAT64
            f['{}-agg-sum'.format(value_feature_name)] = ZERO_FLOAT64
            f['{}-agg-kurtosis'.format(value_feature_name)] = DEFAULT_KURTOSIS_FLOAT64
            f['{}-agg-skewness'.format(value_feature_name)] = ZERO_FLOAT64
    #
    # for key, value in f.items():
    #     print(f'item = {key}, type = {type(value)}')

    return f
