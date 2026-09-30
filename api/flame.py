import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webapp.common import JsonHandler  # noqa: E402
from webapp.flame_backend import run  # noqa: E402


class handler(JsonHandler):
    run = staticmethod(run)
