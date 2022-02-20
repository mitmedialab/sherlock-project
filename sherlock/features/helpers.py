import string
import csv
import io


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


# Notes:
# 1. form feed ('\f') is whitespace but was not classed as such in the original paper, hence not present below.
# 2. '\' and '^' are appended to the list to maintain original column sequence
CHARACTERS_TO_CHECK = (
        [c for c in string.printable if c not in ('\n', '\v', '\r', '\t', '\\', '^')] + ['\\', '^']
)


# Usage:
# from sherlock.features.helpers import generate_chars_col
# generate_chars_col()
def generate_chars_col():
    idx = 0
    with open("../sherlock/features/feature_column_identifiers/char_col.tsv", "w") as char_col:
        for c in CHARACTERS_TO_CHECK:
            for operation in ('any', 'all', 'mean', 'var', 'min', 'max', 'median', 'sum', 'kurtosis', 'skewness'):
                col_header = f'n_{c}-agg-{operation}'

                char_col.write(f'{idx}\t{col_header}\n')

                idx = idx + 1


# Alternative for ast.literal_eval, but keeps the elements as str. This version is about 5x faster than literal_eval
# in this use case
# parse arrays in the form "['a value', None, 0.89, \"other string\"]"
def literal_eval_as_str(value, none_value=None):
    if value and value[0] == '[' and value[-1] == ']':
        value = value[1:-1]

    if not value:
        return []

    strings = []

    quote = None
    s2 = ''

    for s in value.split(', '):
        if not s:
            strings.append('')
        elif s[0] in ["'", '"']:
            if len(s) > 1 and s[0] == s[-1]:
                strings.append(s[1:-1])
            else:
                if quote is None:
                    quote = s[0]
                elif s[0] == quote:
                    s2 = s2 + s[1:]
                    quote = None
                    strings.append(s2 + s[:-1])
                    s2 = ''

                if len(s) == 1:
                    s2 = ', '
                else:
                    s2 = s2 + s[1:] + ', '
        elif quote is not None:
            if quote == s[-1]:
                quote = None
                strings.append(s2 + s[:-1])
                s2 = ''
            else:
                s2 = s2 + s + ', '
        elif s == 'None':
            strings.append(none_value)
        else:
            strings.append(s)

    return strings


def keys_to_csv(keys):
    """
    Encode a list of strings into an Excel CSV compatible header.

    Wraps all items with double quotes to prevent legitimate values containing a comma from being interpreted as a
    separator, and encodes existing double quotes with two double quotes.
    """
    with io.StringIO() as output:
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(keys)

        return output.getvalue()
