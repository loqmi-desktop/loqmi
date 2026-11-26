"""Verify every model URL resolves and its size roughly matches size_gb."""
import json
from pathlib import Path

import httpx

path = Path(__file__).resolve().parents[1] / "model" / "model.json"
data = json.loads(path.read_text(encoding="utf-8"))
models = [m for lst in data.values() for m in lst] if isinstance(data, dict) else data

seen, failed = set(), 0
with httpx.Client(follow_redirects=True, timeout=30) as client:
    for m in models:
        if m["url"] in seen:
            continue
        seen.add(m["url"])
        try:
            with client.stream("GET", m["url"]) as r:       # headers only; the body is never read
                size = int(r.headers.get("content-length", 0))
                status = r.status_code
        except Exception as e:
            print(f"FAIL  {m['id']}: {type(e).__name__}: {e}")
            failed += 1
            continue
        expected = m.get("size_gb", 0) * 1024 ** 3
        ok = status == 200 and (not expected or 0.88 < size / expected < 1.12)
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {m['id']}: HTTP {status}, {size / 1024**3:.2f} GiB (catalog says {m.get('size_gb')})")

raise SystemExit(1 if failed else 0)