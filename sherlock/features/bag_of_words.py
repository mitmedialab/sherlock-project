import math
import nltk
import numpy as np
from scipy.stats import skew, kurtosis
from collections import OrderedDict


ZERO_FLOAT64 = np.float64(0)
DEFAULT_KURTOSIS_FLOAT64 = np.float64(-3.0)


# Input: a single column in the form of a pandas series
# Output: ordered dictionary holding bag of words features
def extract_bag_of_words_features(data_no_null, n_val):
    
    f = OrderedDict()
    
    if not n_val:
        return
    
    # Entropy of column
    freq_dist = nltk.FreqDist(data_no_null)
    probs = [freq_dist.freq(l) for l in freq_dist]
    f['col_entropy'] = -sum(p * math.log(p, 2) for p in probs)

    # Fraction of cells with unique content
    num_unique = data_no_null.nunique()
    f['frac_unique'] = num_unique / n_val

    # Fraction of cells with numeric content -> frac text cells doesn't add information
    num_cells = np.sum(data_no_null.str.contains('[0-9]', regex=True))
    text_cells = np.sum(data_no_null.str.contains('[a-z]|[A-Z]', regex=True))
    f['frac_numcells']  = num_cells / n_val
    f['frac_textcells'] = text_cells / n_val
    
    # Average + std number of numeric tokens in cells
    num_reg = '[0-9]'
    num_result = data_no_null.str.count(num_reg)

    f['avg_num_cells'] = np.mean(num_result)
    f['std_num_cells'] = np.std(num_result)

    # Average + std number of textual tokens in cells
    text_reg = '[a-z]|[A-Z]'
    text_result = data_no_null.str.count(text_reg)

    f['avg_text_cells'] = np.mean(text_result)
    f['std_text_cells'] = np.std(text_result)

    # Average + std number of special characters in each cell
    spec_reg = '[[!@#$%^&*(),.?":{}|<>]]'
    spec_result = data_no_null.str.count(spec_reg)

    f['avg_spec_cells'] = np.mean(spec_result)
    f['std_spec_cells'] = np.std(spec_result)

    # Average number of words in each cell
    space_reg = '[" "]'
    space_result = data_no_null.str.count(space_reg) + 1

    f['avg_word_cells'] = np.mean(space_result)
    f['std_word_cells'] = np.std(space_result)

    all_value_features = OrderedDict()

    f['n_values'] = n_val

    all_value_features['length'] = data_no_null.apply(len)

    for value_feature_name, value_features in all_value_features.items():
        has_any = any(value_features)

        if has_any:
            f[value_feature_name + '-agg-any'] = has_any
            f[value_feature_name + '-agg-all'] = all(value_features)
            f[value_feature_name + '-agg-mean'] = np.float64(np.mean(value_features))
            f[value_feature_name + '-agg-var'] = np.float64(np.var(value_features))
            f[value_feature_name + '-agg-min'] = np.float64(np.min(value_features))
            f[value_feature_name + '-agg-max'] = np.float64(np.max(value_features))
            f[value_feature_name + '-agg-median'] = np.float64(np.median(value_features))
            f[value_feature_name + '-agg-sum'] = np.float64(np.sum(value_features))
            f[value_feature_name + '-agg-kurtosis'] = np.float64(kurtosis(value_features))
            f[value_feature_name + '-agg-skewness'] = np.float64(skew(value_features))
        else:
            f[value_feature_name + '-agg-any'] = False
            f[value_feature_name + '-agg-all'] = False
            f[value_feature_name + '-agg-mean'] = ZERO_FLOAT64
            f[value_feature_name + '-agg-var'] = ZERO_FLOAT64
            f[value_feature_name + '-agg-min'] = ZERO_FLOAT64
            f[value_feature_name + '-agg-max'] = ZERO_FLOAT64
            f[value_feature_name + '-agg-median'] = ZERO_FLOAT64
            f[value_feature_name + '-agg-sum'] = ZERO_FLOAT64
            f[value_feature_name + '-agg-kurtosis'] = DEFAULT_KURTOSIS_FLOAT64
            f[value_feature_name + '-agg-skewness'] = ZERO_FLOAT64

    n_none = data_no_null.size - data_no_null.size - len([e for e in data_no_null if e == ''])
    f['none-agg-has'] = n_none > 0
    f['none-agg-percent'] = n_none / len(data_no_null)
    f['none-agg-num'] = n_none
    f['none-agg-all'] = (n_none == len(data_no_null))
    
    return f
