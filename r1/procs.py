"""Is the process named in a lock file really the one that wrote it?

A lock file outlives its process when Windows shuts down or a run is killed,
and Windows reuses process ids quickly. A bare "is pid N alive" check then
finds some unrelated program under the old id: the launcher refuses to start
training for ever, and the auto-restart watcher could even kill that program
as a hung run. So a lock only counts if its process is Python and was started
no later than the lock was written.
"""
from __future__ import annotations

import os
from pathlib import Path

STILL_ACTIVE = 259


def process_info(pid: int) -> dict | None:
    """{'exe': path or '', 'created': epoch seconds or None}, or None if no
    such process is running."""
    if pid <= 0:
        return None
    if os.name == "nt":
        return _info_windows(pid)
    return _info_posix(pid)


def _info_windows(pid: int) -> dict | None:
    import ctypes
    from ctypes import wintypes
    # A private handle to kernel32, so the argtypes set here cannot clash
    # with other code using ctypes.windll.kernel32.
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE,
                                     ctypes.POINTER(wintypes.DWORD)]
    k.GetProcessTimes.argtypes = [wintypes.HANDLE] + \
        [ctypes.POINTER(wintypes.FILETIME)] * 4
    k.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD)]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k.OpenProcess(0x1000, False, pid)      # QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        code = wintypes.DWORD()
        if not k.GetExitCodeProcess(h, ctypes.byref(code)) or \
                code.value != STILL_ACTIVE:
            return None
        info = {"exe": "", "created": None}
        c, e, kt, ut = (wintypes.FILETIME() for _ in range(4))
        if k.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e),
                             ctypes.byref(kt), ctypes.byref(ut)):
            ticks = (c.dwHighDateTime << 32) | c.dwLowDateTime
            info["created"] = ticks / 1e7 - 11644473600.0
        buf = ctypes.create_unicode_buffer(32768)
        n = wintypes.DWORD(32768)
        if k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
            info["exe"] = buf.value
        return info
    finally:
        k.CloseHandle(h)


def _info_posix(pid: int) -> dict | None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    info = {"exe": "", "created": None}
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        if fields[0] == "Z":                      # exited, not yet reaped
            return None
        with open("/proc/stat") as fh:
            btime = next(int(l.split()[1]) for l in fh if l.startswith("btime"))
        info["created"] = btime + int(fields[19]) / os.sysconf("SC_CLK_TCK")
    except Exception:
        pass
    try:
        info["exe"] = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        pass
    return info


def lock_holder(lock: Path, slack: float = 10.0) -> int | None:
    """The pid of the live Python process that wrote `lock`, else None."""
    try:
        pid = int(lock.read_text().split()[0])
        written = lock.stat().st_mtime
    except (OSError, ValueError, IndexError):
        return None
    info = process_info(pid)
    if info is None:
        return None
    exe = os.path.basename(info.get("exe") or "").lower()
    if exe and "python" not in exe:
        return None                    # the id now belongs to another program
    created = info.get("created")
    if created is not None and created > written + slack:
        return None                    # started after the lock was written
    return pid
