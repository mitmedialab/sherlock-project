import math
import nltk
import numpy as np
from scipy.stats import skew, kurtosis
from collections import OrderedDict
import pandas as pd


ZERO_FLOAT64 = np.float64(0)
DEFAULT_KURTOSIS_FLOAT64 = np.float64(-3.0)


def extract_bag_of_words_features(series: pd.Series, n_val):
    features = OrderedDict()

    extract_bag_of_words_features(series, features, n_val)

    return features


# Input: a single column in the form of a pandas series
# Output: ordered dictionary holding bag of words features
def extract_bag_of_words_features(series: pd.Series, features: OrderedDict, n_val):

    if not n_val:
        return
    
    # Entropy of column
    freq_dist = nltk.FreqDist(series)
    probs = [freq_dist.freq(l) for l in freq_dist]
    features['col_entropy'] = -sum(p * math.log(p, 2) for p in probs)

    # Fraction of cells with unique content
    num_unique = series.nunique()
    features['frac_unique'] = num_unique / n_val

    # Fraction of cells with numeric content -> frac text cells doesn't add information
    num_cells = np.sum(series.str.contains('[0-9]', regex=True))
    text_cells = np.sum(series.str.contains('[a-zA-Z]', regex=True))
    features['frac_numcells'] = num_cells / n_val
    features['frac_textcells'] = text_cells / n_val
    
    # Average + std number of numeric tokens in cells
    num_reg = '[0-9]'
    num_result = series.str.count(num_reg)

    features['avg_num_cells'] = np.mean(num_result)
    features['std_num_cells'] = np.std(num_result)

    # Average + std number of textual tokens in cells
    text_reg = '[a-zA-Z]'
    text_result = series.str.count(text_reg)

    features['avg_text_cells'] = np.mean(text_result)
    features['std_text_cells'] = np.std(text_result)

    # Average + std number of special characters in each cell
    spec_reg = '[\\[!@#$%\\^&*(),.?":{}|<>\\]]'
    spec_result = series.str.count(spec_reg)

    features['avg_spec_cells'] = np.mean(spec_result)
    features['std_spec_cells'] = np.std(spec_result)

    # Average number of words in each cell
    space_reg = '[\\s]'
    space_result = series.str.count(space_reg) + 1

    features['avg_word_cells'] = np.mean(space_result)
    features['std_word_cells'] = np.std(space_result)

    all_value_features = OrderedDict()

    features['n_values'] = n_val

    all_value_features['length'] = series.apply(len)

    for value_feature_name, value_features in all_value_features.items():
        has_any = any(value_features)

        if has_any:
            features[value_feature_name + '-agg-any'] = has_any
            features[value_feature_name + '-agg-all'] = all(value_features)
            features[value_feature_name + '-agg-mean'] = np.mean(value_features)
            features[value_feature_name + '-agg-var'] = np.var(value_features)
            features[value_feature_name + '-agg-min'] = np.min(value_features)
            features[value_feature_name + '-agg-max'] = np.max(value_features)
            features[value_feature_name + '-agg-median'] = np.median(value_features)
            features[value_feature_name + '-agg-sum'] = np.sum(value_features)
            features[value_feature_name + '-agg-kurtosis'] = kurtosis(value_features)
            features[value_feature_name + '-agg-skewness'] = skew(value_features)
        else:
            features[value_feature_name + '-agg-any'] = False
            features[value_feature_name + '-agg-all'] = False
            features[value_feature_name + '-agg-mean'] = ZERO_FLOAT64
            features[value_feature_name + '-agg-var'] = ZERO_FLOAT64
            features[value_feature_name + '-agg-min'] = 0
            features[value_feature_name + '-agg-max'] = 0
            features[value_feature_name + '-agg-median'] = 0
            features[value_feature_name + '-agg-sum'] = 0
            features[value_feature_name + '-agg-kurtosis'] = DEFAULT_KURTOSIS_FLOAT64
            features[value_feature_name + '-agg-skewness'] = ZERO_FLOAT64

    n_none = series.size - series.size - len([e for e in series if e == ''])
    features['none-agg-has'] = n_none > 0
    features['none-agg-percent'] = n_none / len(series)
    features['none-agg-num'] = n_none
    features['none-agg-all'] = (n_none == len(series))
