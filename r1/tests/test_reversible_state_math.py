import numpy as np


def test_ternary_rule_selection_and_page_transitions_are_one_to_one():
    states = np.arange(243)
    powers = 3 ** np.arange(5)
    trits = (states[:, None] // powers[None, :]) % 3 - 1
    values = 127 * trits
    bias = -64 * (trits != 0).sum(axis=1)
    scores = values @ trits.T + bias[None, :]
    assert np.array_equal(scores.argmax(axis=1), states)
    for multiplier in (1, 2, 4, 5, 7, 10):
        for page in range(32):
            next_state = (multiplier * states + page + 1) % 243
            assert len(np.unique(next_state)) == 243
