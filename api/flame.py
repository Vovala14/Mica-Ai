import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webapp.common import make_handler  # noqa: E402
from webapp.flame_backend import run  # noqa: E402

handler = make_handler(run)
