"""
core/lifecycle.py
Exits the backend if the desktop app that spawned it disappears
(force quit, crash, Ctrl+C on `tauri dev`), which no Rust exit handler can catch.
Only active when the app sets loqmi_PARENT_WATCH=1, so running
`uvicorn` by hand in a terminal is unaffected.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time

logger = logging.getLogger(__name__)


def start_parent_watchdog(poll_seconds: float = 1.0, grace_seconds: float = 5.0) -> None:
    if os.environ.get("loqmi_PARENT_WATCH") != "1":
        return

    original_parent = os.getppid()

    def _watch() -> None:
        # ppid changes (to 1/launchd) when the parent dies.
        while os.getppid() == original_parent and original_parent != 1:
            time.sleep(poll_seconds)
        logger.warning("Parent app is gone; shutting down backend")
        os.kill(os.getpid(), signal.SIGTERM)   # let uvicorn shut down cleanly
        time.sleep(grace_seconds)
        os._exit(0)                            # hard exit if graceful shutdown hangs

    threading.Thread(target=_watch, name="parent-watchdog", daemon=True).start()