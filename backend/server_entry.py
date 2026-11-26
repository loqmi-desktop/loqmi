import sys
import os
import multiprocessing
from pathlib import Path

if getattr(sys, 'frozen', False):
    bundle_dir = Path(sys._MEIPASS)
    sys.path.insert(0, str(bundle_dir))
    os.chdir(bundle_dir)

    # Frozen apps don't see the system CA store reliably; point TLS at certifi's bundle.
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import uvicorn

from config import API_HOST, API_PORT

if __name__ == "__main__":
    multiprocessing.freeze_support()   # must run first in __main__ on Windows
    if "--selftest" in sys.argv:
        try:
            from core import classifier
            from rag.embeddings import get_embeddings
            print("classifier:", classifier.classify_with_confidence("hello"))
            print("embeddings: OK, dim =", len(get_embeddings().embed_query("hello world")))
            print("SELFTEST OK")
            sys.exit(0)
        except Exception:
            import traceback
            traceback.print_exc()
            print("SELFTEST FAILED")
            sys.exit(1)
    uvicorn.run(
        "api.server:app",
        host=API_HOST,
        port=API_PORT,
        log_level="info",
    )