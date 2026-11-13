"""
api/server.py — FastAPI application entry point
Mounts all routers and handles startup/shutdown lifecycle.
Tauri spawns this as a sidecar process on app launch.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import memory, model_manager
from api.routes import (
    chats,
    documents,
    setup,
    attachements,
    context_window,
    chat_documents,
)
from core.lifecycle import start_parent_watchdog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for _noisy in ("primp", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

start_parent_watchdog()


def _flatten_catalog(catalog_data) -> list[dict]:
    """model.json is either a flat list of models or an object keyed by category."""
    if isinstance(catalog_data, list):
        return catalog_data
    flat: list[dict] = []
    for category, model_list in catalog_data.items():
        for m in model_list:
            m["category"] = category
            flat.append(m)
    return flat


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup — initialise DB
    memory.init_db()

    # 2. Auto-seed catalog if empty
    if not memory.get_suggested_models():
        logger.info("Catalog is empty. Auto-seeding initial models...")
        catalog_path = Path(__file__).resolve().parents[2] / "model" / "model.json"

        try:
            if catalog_path.exists():
                with open(catalog_path, "r", encoding="utf-8") as f:
                    flat_list = _flatten_catalog(json.load(f))
                memory.sync_suggested_models(flat_list)
                logger.info("Successfully seeded %d models.", len(flat_list))
            else:
                logger.warning(
                    "Seed file not found at %s. The catalog stays empty until "
                    "POST /setup/refresh-catalog succeeds. (In a PyInstaller build, "
                    "make sure model/model.json is listed in the spec's datas.)",
                    catalog_path,
                )
        except Exception as e:
            logger.error("Failed to auto-seed catalog: %s", e)

    logger.info("loqmi API ready")
    yield

    # Shutdown — unload any model held in RAM
    model_manager.unload()
    logger.info("Shutdown complete")


app = FastAPI(
    title="loqmi API",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",
        "https://tauri.localhost",
        "http://localhost:1420",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chats.router)
app.include_router(chat_documents.router)
app.include_router(documents.router)
app.include_router(setup.router)
app.include_router(attachements.router)
app.include_router(context_window.router)

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=False)