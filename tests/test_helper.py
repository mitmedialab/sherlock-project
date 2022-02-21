from sherlock.deploy.helpers import categorize_features
from collections import OrderedDict

CATEGORY_FEATURE_KEYS: dict = categorize_features()


def assert_category_keys(category: str, features: OrderedDict):
    assert category in CATEGORY_FEATURE_KEYS

    expected_keys = CATEGORY_FEATURE_KEYS[category]

    keys = set(expected_keys)

    # No duplicates in the expected key list
    assert len(keys) == len(expected_keys)

    # All feature keys are accounted for in the expected char feature list...
    for f in expected_keys:
        assert f in features, f'Failed to find key "{f}"'

    # ... and no extras
    for k in features.keys():
        keys.remove(k)

    assert len(keys) == 0
