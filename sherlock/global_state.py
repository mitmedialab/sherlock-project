_is_first = True


def is_first() -> bool:
    return _is_first


def set_first():
    global _is_first
    _is_first = False


def reset_first():
    global _is_first
    _is_first = True
