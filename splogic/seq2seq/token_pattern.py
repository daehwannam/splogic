
import re


def is_bart_digit_seq_token(token_value):
    return _bart_digit_seq_token_regex.match(token_value)


_bart_digit_seq_token_regex = re.compile(r'^Ġ?[0-9]+$')
