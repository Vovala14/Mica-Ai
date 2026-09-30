#!/usr/bin/env python3
"""Auto-restart watcher for MICA training.

Started once by INSTALL_AUTORESTART.bat, then again automatically at every
Windows logon. Its one job is to keep mica/r1/run_train.py running unless it
has been told to stop. It never starts anything else.

Controls are plain files in PycharmProjects, which Claude can also create
remotely:

  STOP_MICA        training stops at the next step (state saved) and STAYS
                   stopped. START_TRAIN.bat removes it and starts again.
  RESTART_MICA     training stops cleanly, then starts again straight away
                   with whatever code is in the folder now. The file is
                   removed once that is done.
  SUPERVISOR_EXIT  this watcher exits. Training is left exactly as it is.

What it does on its own:

  * starts training whenever it is not running, unless it was stopped or has
    finished all its steps
  * after an unexpected exit, waits 5 minutes and starts it again; if it
    keeps failing the wait doubles, up to an hour, so a broken setup cannot
    thrash the PC
  * if the trainer's heartbeat (r1/runs/soft/heartbeat) has been silent for
    45 minutes, the run is hung: it is killed and started again, and resumes
    from its last save (at most 50 steps lost). stall_trace.txt in the same
    folder records where it hung.

Everything it does goes into mica_supervisor.log, and mica_supervisor.alive
is rewritten on every check, so its state can be read remotely.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # PycharmProjects/mica/r1
sys.path.insert(0, str(HERE))
from procs import lock_holder                   # noqa: E402

ROOT = HERE.parents[1]                          # PycharmProjects
RUNS = HERE / "runs" / "soft"
LAUNCHER = HERE / "run_train.py"
TRAINER = HERE / "train_soft.py"
LOG = ROOT / "mica_supervisor.log"
ALIVE = ROOT / "mica_supervisor.alive"
OWN_LOCK = ROOT / "mica_supervisor.lock"
TRAIN_LOCK = ROOT / "mica_train.lock"
STOP = ROOT / "STOP_MICA"
RESTART = ROOT / "RESTART_MICA"
EXIT = ROOT / "SUPERVISOR_EXIT"
HEARTBEAT = RUNS / "heartbeat"
FINISHED = RUNS / "FINISHED"
CHILD_ERRORS = ROOT / "mica_supervisor_child.err"


def _env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# Timings, in seconds. The environment overrides exist for the tests.
POLL = _env("MICA_SUP_POLL", 30)
STALL = _env("MICA_SUP_STALL", 45 * 60)          # heartbeat silence = hung
BACKOFF = _env("MICA_SUP_BACKOFF", 5 * 60)       # first wait after a failure
BACKOFF_MAX = _env("MICA_SUP_BACKOFF_MAX", 60 * 60)
STOP_TIMEOUT = _env("MICA_SUP_STOP_TIMEOUT", 40 * 60)   # clean stop too slow
HEALTHY = _env("MICA_SUP_HEALTHY", 60 * 60)      # a run this long resets it

WINDOWS = os.name == "nt"
BELOW_NORMAL, NEW_GROUP, NO_WINDOW = 0x00004000, 0x00000200, 0x08000000


def say(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _rm(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        say(f"could not remove {p.name}: {exc}")


def take_own_lock() -> bool:
    for _ in range(3):
        try:
            fd = os.open(str(OWN_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if lock_holder(OWN_LOCK):
                return False                   # a watcher is already running
            _rm(OWN_LOCK)
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    return False


def release_own_lock() -> None:
    try:
        if OWN_LOCK.read_text().split()[0] == str(os.getpid()):
            OWN_LOCK.unlink()
    except (OSError, IndexError):
        pass


def trainer_is_safe() -> bool:
    """The same check START_TRAIN.bat makes: never launch the old trainer
    that could freeze the PC."""
    try:
        return LAUNCHER.exists() and \
            "set_per_process_memory_fraction" in TRAINER.read_text("utf-8")
    except OSError:
        return False


def launch() -> subprocess.Popen:
    err = open(CHILD_ERRORS, "a", encoding="utf-8")
    try:
        kw = {"creationflags": BELOW_NORMAL | NEW_GROUP | NO_WINDOW} \
            if WINDOWS else {"start_new_session": True}
        return subprocess.Popen([sys.executable, str(LAUNCHER)], cwd=str(ROOT),
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=err, **kw)
    finally:
        err.close()


def kill_tree(pid: int) -> None:
    """Kill the launcher and the trainer under it. Only ever called with the
    pid from the training lock, after lock_holder() has vouched for it."""
    if WINDOWS:
        r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, text=True,
                           creationflags=NO_WINDOW)
        say(f"taskkill: {(r.stdout or r.stderr).strip()[:200]}")
        return
    import signal
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def write_alive(status: str) -> None:
    try:
        ALIVE.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {status}\n"
                         f"pid {os.getpid()}\n", encoding="utf-8")
    except OSError:
        pass


def main() -> int:
    if not take_own_lock():
        return 0
    say(f"watcher started (pid {os.getpid()}); checking every {POLL:.0f} s")
    child = None            # our Popen of run_train.py, if we started it
    was_running = False
    started_at = None       # when the current run started
    wait = BACKOFF          # next wait after a failure
    not_before = 0.0        # no automatic start before this time
    restart_since = None    # a RESTART_MICA is being carried out
    last_status = None
    try:
        while True:
            now = time.time()
            if EXIT.exists():
                _rm(EXIT)
                say("SUPERVISOR_EXIT found: watcher exiting, training left "
                    "as it is")
                break
            if child is not None and child.poll() is not None:
                child = None
            pid = lock_holder(TRAIN_LOCK)
            running = pid is not None or child is not None

            # ---- a run has just ended
            if was_running and not running:
                lasted = now - (started_at or now)
                if restart_since is not None:
                    say("training stopped for the restart")
                elif STOP.exists() or FINISHED.exists():
                    pass                   # the status line below says which
                else:
                    if lasted >= HEALTHY:
                        wait = BACKOFF
                    not_before = now + wait
                    say(f"training ended unexpectedly after "
                        f"{lasted / 60:.0f} min; starting it again in "
                        f"{wait / 60:.0f} min")
                    wait = min(wait * 2, BACKOFF_MAX)
                started_at = None
            if running and not was_running:
                started_at = _mtime(TRAIN_LOCK) or now
                if child is None:
                    say(f"training is running (pid {pid}), started outside "
                        f"the watcher")
            was_running = running

            # ---- a restart was requested
            if RESTART.exists() and restart_since is None:
                restart_since, wait, not_before = now, BACKOFF, 0.0
                _rm(FINISHED)
                if running:
                    STOP.touch()
                    say("RESTART_MICA: asking training to stop cleanly, then "
                        "starting it again")
                else:
                    say("RESTART_MICA: training is not running; starting it")
            if restart_since is not None:
                if running:
                    if pid and now - restart_since > STOP_TIMEOUT:
                        say("the clean stop is taking too long: killing the "
                            "run (it resumes from its last save)")
                        kill_tree(pid)
                    write_alive("restarting: waiting for training to stop")
                    time.sleep(POLL)
                    continue
                _rm(STOP)
                _rm(RESTART)
                restart_since, not_before = None, 0.0

            # ---- a hung run
            if running and pid:
                quiet = now - max(started_at or 0.0, _mtime(HEARTBEAT),
                                  _mtime(TRAIN_LOCK))
                if quiet > STALL:
                    say(f"no heartbeat for {quiet / 60:.0f} min: the run is "
                        f"hung. Killing it; it restarts from its last save.")
                    kill_tree(pid)

            # ---- start it if it should be running
            if running:
                status = f"training running (pid {pid or child.pid})"
            elif STOP.exists():
                status = ("training stopped by STOP_MICA; it stays stopped "
                          "until START_TRAIN.bat or RESTART_MICA")
            elif FINISHED.exists():
                status = "training finished all its steps; not restarting"
            elif now < not_before:
                status = (f"waiting to start again at "
                          f"{time.strftime('%H:%M', time.localtime(not_before))}")
            elif not trainer_is_safe():
                status = ("not starting: run_train.py missing or train_soft.py "
                          "lacks the memory protections")
            else:
                child = launch()
                started_at, was_running = now, True
                status = f"started training (pid {child.pid})"
                say(status)
            quiet_kinds = ("training running", "started", "waiting")
            if status != last_status and not status.startswith(quiet_kinds):
                say(status)          # the others were logged where they began
            last_status = status
            write_alive(status)
            time.sleep(POLL)
    except KeyboardInterrupt:
        say("watcher interrupted")
    finally:
        release_own_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
