"""The topic register (spec.TOPIC_CHANNELS): exact arithmetic, the engine and
the fitter's simulator agree, conversion from a model without it changes only
what it should, and topic and plain files refuse to load as each other."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _topic_cases as C  # noqa: E402


def run(env: dict, *args) -> str:
    e = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    e.update(env)
    r = subprocess.run([sys.executable, str(HERE / "_topic_cases.py"), *args],
                       env=e, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("topic")
    base, topic = d / "base.mica", d / "topic.mica"
    run(C.BASE, "make_base", str(base))
    assert run(C.TOPIC, "topic", str(base), str(topic)) == "ok"
    return base, topic


def test_engine_and_simulator_agree(files):
    pass                                         # asserted inside the fixture's case


def test_topic_file_refused_without_register(files):
    assert run(C.BASE, "refuse", str(files[1])) == "refused"


def test_plain_file_refused_with_register(files):
    assert run(C.TOPIC, "refuse", str(files[0])) == "refused"


def test_register_off_by_default():
    from mica_r1 import spec
    assert spec.TOPIC_CHANNELS == 0 and spec.TOPIC_AT == spec.N_CHANNELS
