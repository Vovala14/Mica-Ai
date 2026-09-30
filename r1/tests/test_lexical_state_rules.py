"""The integer word-state rule preserves the last word across separators."""
from __future__ import annotations

from mica_r1.fit import lexical_state_transition


def test_previous_word_survives_punctuation_and_space() -> None:
    previous, current = 0, 121  # BOS is a separator
    for page in (3, 5, 2):  # letters of one word
        previous, current = (
            lexical_state_transition(previous, page, True, 0),
            lexical_state_transition(current, page, True, 1))
    first_word = current
    assert 0 <= first_word < 121
    for page in (0, 0):  # comma, then a space
        previous, current = (
            lexical_state_transition(current, page, False, 0),
            lexical_state_transition(current, page, False, 1))
        assert previous == first_word
        assert current == 121 + first_word
    previous, current = (
        lexical_state_transition(previous, 4, True, 0),
        lexical_state_transition(current, 4, True, 1))
    assert previous == first_word
    assert current == 5  # the new word starts from its first byte class


def test_letter_routes_are_distinct_and_word_pages_disjoint(monkeypatch) -> None:
    import torch
    import mica_r1.fit as fit

    monkeypatch.setattr(fit, "ROUTING_CHANNELS", (0, 1, 2, 3, 4))
    monkeypatch.setattr(fit, "ROUTING_PAIR_OFFSET", 5)
    code = fit.reserve_letter_routes(torch.zeros(258, 16))

    def route(byte: int) -> int:
        return sum(int(code[byte, i] >= code[byte, i + 5]) << i
                   for i in range(5))

    assert [route(byte) for byte in range(97, 123)] == list(range(1, 27))
    assert [route(byte) for byte in range(65, 91)] == list(range(1, 27))
    assert route(ord(" ")) == 0
    assert route(ord("0")) == 27
    assert route(ord("'")) == 28
    assert route(200) == 29
