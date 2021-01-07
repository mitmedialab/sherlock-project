import numpy as np


# Good explanation of Kurtosis and Skewness
# https://www.youtube.com/watch?v=lK7nLzxiAQQ
#
# based on code found here:
# https://stackoverflow.com/questions/61521371/calculate-weighted-statistical-moments-in-python
def compute_stats(values):
    _min = min(values)
    _max = max(values)
    _sum = sum(values)

    _mean = np.mean(values)

    x = values - _mean

    _variance = np.mean(x * x)

    if _variance == 0:
        _skew = 0
        _kurtosis = -3
    else:
        _skew = np.mean(x ** 3) / _variance ** 1.5
        _kurtosis = np.mean(x ** 4) / _variance ** 2 - 3

    return _mean, _variance, _skew, _kurtosis, _min, _max, _sum


def mode(axis, pre_sorted: bool = False):
    if not pre_sorted:
        axis = sorted(axis)

    _count_max = 1
    _count = 0
    _mode = _current = axis[0]

    for v in axis:
        if v == _current:
            _count = _count + 1
        else:
            if _count > _count_max:
                _count_max = _count
                _mode = _current
            _count = 1
            _current = v

    if _count > _count_max:
        return _current

    return _mode
