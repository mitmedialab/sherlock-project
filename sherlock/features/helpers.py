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
        ['[' + escape_for_regex(c) + ']' for c in string.printable if c not in ('\n', '\f', '\v', '\r', '\t')]
)


def generate_chars_col():
    idx = 0
    with open("../sherlock/features/feature_column_identifiers/char_col.tsv", "w") as char_col:
        for c in CHARACTERS_TO_CHECK:
            for operation in ('any', 'all', 'mean', 'var', 'min', 'max', 'median', 'sum', 'kurtosis', 'skewness'):
                col_header = f'n_{c}-agg-{operation}'

                char_col.write(f'{idx}\t{col_header}\n')

                idx = idx + 1
