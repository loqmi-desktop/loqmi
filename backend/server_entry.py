"""
server_entry.py — PyInstaller entry point for the loqmi backend sidecar.
Tauri spawns this binary directly. It starts uvicorn on 127.0.0.1:8765.
"""
import sys
import os
from pathlib import Path

# When frozen by PyInstaller, sys._MEIPASS is the temp dir with bundled files.
# We need it on the path so all backend modules (config, api, core, rag) import correctly.
if getattr(sys, 'frozen', False):
    bundle_dir = Path(sys._MEIPASS)
    sys.path.insert(0, str(bundle_dir))
    os.chdir(bundle_dir)

import uvicorn

from config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host=API_HOST,
        port=API_PORT,
        log_level="info",
    )