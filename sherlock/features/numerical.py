from collections import OrderedDict
import re


NUMBER_PATTERN = re.compile(r'[0-9]')


def extract_numeric(text):
    m = re.search(NUMBER_PATTERN, text)

    if m:
        return m.group(1)
    else:
        return None


# Input: a single column in the form of a Python list
# Output: add numerical features to the ordered dictionary
def extract_numerical_collection_features(col_values: list, features: OrderedDict, n_val):

    numerical_values = sorted(filter(lambda n: n is not None, map(lambda s: extract_numeric(s), col_values)))

    return None
