import string
import numpy  as np
from scipy.stats import skew, kurtosis
from collections import OrderedDict


# Input: a single column in the form of pandas series
# Output: ordered dictionary holding bag of character features
def extract_bag_of_characters_features(data):
    
    characters_to_check = (
            ['['+ c + ']' for c in string.printable if c not in ('\n', '\\', '\v', '\r', '\t', '^')]
            + ['[\\\\]', '[\^]']
    )
    
    f = OrderedDict()

    data_no_null = data.dropna()
    all_value_features = OrderedDict()

    for c in characters_to_check:
        all_value_features['n_{}'.format(c)] = data_no_null.str.count(c)
        
    for value_feature_name, value_features in all_value_features.items():
        f['{}-agg-any'.format(value_feature_name)] = any(value_features)
        f['{}-agg-all'.format(value_feature_name)] = all(value_features)
        f['{}-agg-mean'.format(value_feature_name)] = np.mean(value_features)
        f['{}-agg-var'.format(value_feature_name)] = np.var(value_features)
        f['{}-agg-min'.format(value_feature_name)] = np.min(value_features)
        f['{}-agg-max'.format(value_feature_name)] = np.max(value_features)
        f['{}-agg-median'.format(value_feature_name)] = np.median(value_features)
        f['{}-agg-sum'.format(value_feature_name)] = np.sum(value_features)
        f['{}-agg-kurtosis'.format(value_feature_name)] = kurtosis(value_features)
        f['{}-agg-skewness'.format(value_feature_name)] = skew(value_features)

    return f


    
    

