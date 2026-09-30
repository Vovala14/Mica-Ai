"""Changing a rule book at resume must never silently overwrite a best model."""
from types import SimpleNamespace

from train_soft import (INIT_KEYS, LEGACY_INIT_DEFAULTS,
                        initialization_signature, resume_problem)


def test_initialization_options_checked_for_new_and_legacy_checkpoints():
    args = SimpleNamespace(**{key: "original" for key in INIT_KEYS})
    initial = initialization_signature(args)
    signature = list(range(12))
    assert resume_problem(signature, signature, initial, initial) is None
    assert resume_problem(signature, signature, None, initial, initial) is None

    changed = dict(initial, rule_max_back="1,2,2,3")
    assert "rule_max_back" in resume_problem(
        signature, signature, initial, changed)
    assert "rule_max_back" in resume_problem(
        signature, signature, None, changed, initial)
    assert "cannot be verified" in resume_problem(
        signature, signature, None, initial, None)


def test_resume_rejects_changed_geometry_but_allows_longer_run():
    initial = {key: "same" for key in INIT_KEYS}
    signature = list(range(12))
    extended = signature.copy()
    extended[9] += 10
    assert resume_problem(signature, extended, initial, initial) is None
    changed = signature.copy()
    changed[0] += 1
    assert "geometry" in resume_problem(
        signature, changed, initial, initial)


def test_older_run_info_uses_historical_defaults_only_when_missing():
    initial = {key: "same" for key in INIT_KEYS}
    initial.update(LEGACY_INIT_DEFAULTS)
    old_args = {key: value for key, value in initial.items()
                if key not in LEGACY_INIT_DEFAULTS}
    signature = list(range(12))
    assert resume_problem(signature, signature, None, initial, old_args) is None
    changed = dict(initial, rule_lag_coverage=True)
    assert "rule_lag_coverage" in resume_problem(
        signature, signature, None, changed, old_args)
