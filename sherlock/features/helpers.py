import string


# https://stackoverflow.com/questions/10593387/when-do-i-need-to-escape-characters-within-a-regex-character-set-within

# Also include '[' char despite above to prevent the following warning:
# .../sherlock-project/sherlock/features/bag_of_characters.py:38: FutureWarning: Possible nested set at position 1
#   if search(c, char_set):
#
# NOTE: Make sure that each item that is escaped has the escaping here. The file may be regenerated using
#       generate_chars_col() below.
#          sherlock/features/feature_column_identifiers/char_col.tsv
def escape_for_regex(c):
    if c in ('[', ']', '\\', '^', '-'):
        return '\\' + c
    else:
        return c


CHARACTERS_TO_CHECK = (
        [c for c in string.printable if c not in ('\n', '\f', '\v', '\r', '\t')]
)


# Usage:
# from sherlock.features.helpers import generate_chars_col
# generate_chars_col()
def generate_chars_col():
    idx = 0
    with open("../sherlock/features/feature_column_identifiers/char_col.tsv", "w") as char_col:
        for c in CHARACTERS_TO_CHECK:
            for operation in ('any', 'all', 'mean', 'var', 'min', 'max', 'median', 'sum', 'kurtosis', 'skewness'):
                col_header = f'n_[{c}]-agg-{operation}'

                char_col.write(f'{idx}\t{col_header}\n')

                idx = idx + 1


# Alternative for ast.literal_eval, but keeps the elements as str. This version is about 5x faster than literal_eval
# in this use case
# parse arrays in the form "['a value', None, 0.89, "other string"]
def literal_eval_as_str(value, none_value=None):
    strings = []

    quote = None
    s2 = ''

    if len(value) == 0:
        return strings

    if value[0] == '[':
        value = value[1:len(value) - 1]

    for s in value.split(', '):
        if len(s) == 0:
            strings.append('')
        elif s[0] in ["'", '"']:
            if s[0] == s[len(s) - 1]:
                strings.append(s[1:len(s) - 1])
            else:
                quote = s[0]
                s2 = s2 + s[1:] + ', '
        elif quote is not None:
            if quote == s[len(s) - 1]:
                quote = None
                strings.append(s2 + s[:len(s) - 1])
                s2 = ''
            else:
                s2 = s2 + s + ', '
        elif s == 'None':
            strings.append(none_value)
        else:
            strings.append(s)

    return strings
