import numpy as np


# Good explanation of Kurtosis and Skewness
# https://www.youtube.com/watch?v=lK7nLzxiAQQ
#
# based on code found here:
# https://stackoverflow.com/questions/61521371/calculate-weighted-statistical-moments-in-python
def compute_stats(values):
    _min = 2147483647
    _max = -2147483648
    _sum = 0

    for v in values:
        if v < _min:
            _min = v
        if v > _max:
            _max = v
        _sum = _sum + v

    _mean = np.mean(values)

    x = values - _mean

    _variance = np.mean(x ** 2)

    if _variance == 0:
        _skew = 0
        _kurtosis = -3
    else:
        _skew = np.mean(x ** 3) / _variance ** 1.5
        _kurtosis = np.mean(x ** 4) / _variance ** 2 - 3

    return _mean, _variance, _skew, _kurtosis, _min, _max, _sum
