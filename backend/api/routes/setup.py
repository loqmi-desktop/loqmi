from __future__ import annotations

import asyncio
import json
from pathlib import Path
import urllib.request

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from huggingface_hub import HfApi

from config import (
    ALL_MODEL_CATEGORIES,
    HELPER_CATEGORY,
    get_models_for_category, model_path, model_exists,
    get_recommended_model, get_model_config,
)
from core import memory, model_manager

router = APIRouter(prefix="/setup", tags=["setup"])

DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk

def get_all_models() -> dict[str, list[dict]]:
    return {
        category: get_models_for_category(category)
        for category in ALL_MODEL_CATEGORIES
    }

# ── Token Management ──────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    token: str

@router.get("/token")
def get_token():
    """Returns the masked HF token if it exists."""
    token = memory.get_hf_token()
    if not token:
        return {"has_token": False, "masked_token": None}
    
    masked = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "****"
    return {"has_token": True, "masked_token": masked}

@router.post("/token")
def set_token(body: TokenRequest):
    """Saves the HF token to the database."""
    clean_token = body.token.strip()
    memory.set_hf_token(clean_token)
    return {"status": "success"}

# ── Status ────────────────────────────────────────────────────────────────────

@router.get("/status")
def get_status():
    """
    Returns download status and rich metadata for every model variant.

    `all_ready` means "the app is usable": a chat model is on disk. The other
    categories (code, reasoning, document) fall back to the chat model, so they
    are optional upgrades and must not keep users on the setup screen.
    """
    status = {}
    seen_files: set[str] = set()

    for category, model_list in get_all_models().items():
        status[category] = []
        
        active_id = memory.get_active_model(category)
        
        for cfg in model_list:
            filename = cfg["filename"]
            model_id = cfg["id"]
            downloaded = model_exists(category, model_id)

            model_info = {
                "id":               model_id,
                "name":             cfg["name"],
                "filename":         filename,
                "downloaded":       downloaded,
                "size_gb":          cfg["size_gb"],
                "n_ctx":            cfg.get("n_ctx"),
                "ram_required":     cfg.get("ram_required"),
                "hardware_target":  cfg.get("hardware_target"),
                "description":      cfg.get("description"),
                "recommended":      cfg.get("recommended", False),
                "is_active":        (model_id == active_id),
            }

            if filename in seen_files:
                model_info["shared_with"] = "chat"
            else:
                seen_files.add(filename)

            status[category].append(model_info)

    all_ready = any(
        model_exists("chat", m["id"]) for m in get_all_models().get("chat", [])
    )
    
    return {"models": status, "all_ready": all_ready}

# ── Catalog & Downloaded Models ──────────────────────────────────────────────

@router.get("/catalog")
def get_catalog():
    """Returns the curated catalog of models stored in SQLite."""
    try:
        return memory.get_suggested_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch catalog: {str(e)}")

@router.post("/refresh-catalog")
def refresh_catalog():
    """
    Refreshes the catalog into SQLite.
    Reads local model/model.json in dev mode, or fetches from GitHub in production.
    """
    catalog_path = (
        Path(__file__).resolve().parents[2]
        / "model"
        / "model.json"
    )

    try:
        if catalog_path.exists():
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog_data = json.load(f)
        else:
            # Production fallback if local file does not exist.
            # TODO before open-sourcing: replace the placeholder with the real hosted catalog URL.
            url = "https://raw.githubusercontent.com/your-username/loqmi-catalog/main/models.json"
            req = urllib.request.Request(url, headers={"User-Agent": "loqmi"})
            with urllib.request.urlopen(req) as response:
                catalog_data = json.loads(response.read().decode("utf-8"))

        # Flatten if dictionary format
        flat_list = []
        if isinstance(catalog_data, dict):
            for category, model_list in catalog_data.items():
                for m in model_list:
                    m["category"] = category
                    flat_list.append(m)
        else:
            flat_list = catalog_data

        memory.sync_suggested_models(flat_list)

        return {
            "status": "success",
            "count": len(flat_list)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model catalog: {str(e)}"
        )

@router.get("/downloaded")
def get_downloaded():
    """Returns models currently downloaded locally."""
    try:
        return memory.get_downloaded_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch downloaded models: {str(e)}")

# ── Hugging Face Live Search ─────────────────────────────────────────────────

@router.get("/hf-search")
def search_huggingface(q: str, limit: int = 10):
    """Searches Hugging Face Hub for GGUF models matching query string."""
    try:
        api = HfApi()
        models = api.list_models(
            search=q, 
            tags=["gguf", "text-generation"], 
            sort="downloads", 
            direction=-1, 
            limit=limit
        )
        
        results = []
        for m in models:
            results.append({
                "repo_id": m.id,
                "downloads": getattr(m, "downloads", 0),
                "available_files": []
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face search failed: {str(e)}")

@router.get("/hf-files")
def get_huggingface_files(repo_id: str):
    """Lists all available .gguf files within a specific Hugging Face repository."""
    try:
        api = HfApi()
        info = api.model_info(repo_id)
        gguf_files = [f.rfilename for f in info.siblings if f.rfilename.endswith(".gguf")]
        return gguf_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch repo files: {str(e)}")

# ── Global Repair Download (Recommended Sync) ─────────────────────────────────

@router.get("/download")
async def download_models():
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def downloader():
        seen_files: set[str] = set()
        
        headers = {}
        hf_token = memory.get_hf_token()
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        async with httpx.AsyncClient(timeout=None, follow_redirects=True, headers=headers) as client:
            for category in get_all_models().keys():
                if category in ("image", HELPER_CATEGORY):
                    continue

                cfg = get_recommended_model(category)
                if not cfg:
                    continue

                filename = cfg["filename"]
                model_id = cfg["id"]
                url      = cfg.get("url")

                if filename in seen_files:
                    await queue.put({"type": "skip", "model": model_id, "reason": "shared resource"})
                    continue

                seen_files.add(filename)
                dest = model_path(category, model_id)

                if model_exists(category, model_id):
                    await queue.put({"type": "skip", "model": model_id, "reason": "already downloaded"})
                    continue

                model_manager.set_download_state(category, model_id, "downloading")

                await queue.put({
                    "type":     "start",
                    "model":    model_id,
                    "filename": filename,
                    "size_gb":  cfg["size_gb"],
                })

                try:
                    async with client.stream("GET", url) as response:
                        if response.status_code in (401, 403):
                            raise Exception("Hugging Face API Token is invalid or expired.")
                        response.raise_for_status()
                        
                        total = int(response.headers.get("content-length", cfg["size_gb"] * 1024 ** 3))
                        downloaded = 0

                        with open(dest, "wb") as f:
                            async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK_SIZE):
                                await asyncio.to_thread(f.write, chunk)
                                downloaded += len(chunk)
                                percent = round((downloaded / total) * 100, 1) if total else 0
                                await queue.put({
                                    "type":          "progress",
                                    "model":         model_id,
                                    "percent":       percent,
                                    "downloaded_gb": round(downloaded / 1024 ** 3, 2),
                                })

                    model_manager.set_download_state(category, model_id, "ready")
                    await queue.put({"type": "done", "model": model_id})

                except Exception as e:
                    dest.unlink(missing_ok=True)
                    model_manager.set_download_state(category, model_id, "not_downloaded")
                    await queue.put({"type": "error", "model": model_id, "message": str(e)})

        await queue.put({"type": "complete"})
        await queue.put(None)

    async def event_stream():
        task = asyncio.create_task(downloader())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── Targeted Single Model Downloader ──────────────────────────────────────────

@router.post("/models/{category}/{model_id}")
async def download_targeted_model(category: str, model_id: str):
    if category not in get_all_models():
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found.")
    
    cfg = get_model_config(category, model_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Model ID '{model_id}' not found in '{category}'.")

    url = cfg.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="This model utilizes shared assets and cannot be downloaded directly.")

    current_state = model_manager.get_download_state(category, model_id)
    if current_state == "downloading":
        raise HTTPException(status_code=409, detail="This model is already actively downloading.")

    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def single_downloader():
        dest = model_path(category, model_id)
        model_manager.set_download_state(category, model_id, "downloading")

        await queue.put({
            "type":     "start",
            "model":    model_id,
            "filename": cfg["filename"],
            "size_gb":  cfg["size_gb"],
        })

        try:
            headers = {}
            hf_token = memory.get_hf_token()
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"

            async with httpx.AsyncClient(timeout=None, follow_redirects=True, headers=headers) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code in (401, 403):
                        raise Exception("Hugging Face API Token is invalid or expired.")
                    response.raise_for_status()
                    
                    total = int(response.headers.get("content-length", cfg["size_gb"] * 1024 ** 3))
                    downloaded = 0

                    with open(dest, "wb") as f:
                        async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK_SIZE):
                            await asyncio.to_thread(f.write, chunk)
                            downloaded += len(chunk)
                            percent = round((downloaded / total) * 100, 1) if total else 0
                            await queue.put({
                                "type":          "progress",
                                "model":         model_id,
                                "percent":       percent,
                                "downloaded_gb": round(downloaded / 1024 ** 3, 2),
                            })

            model_manager.set_download_state(category, model_id, "ready")
            memory.set_active_model(category, model_id)
            
            # Register in downloaded_models table
            memory.add_downloaded_model({
                "id": model_id,
                "category": category,
                "name": cfg["name"],
                "filename": cfg["filename"],
                "size_gb": cfg.get("size_gb"),
                "n_ctx": cfg.get("n_ctx", 8192),
                "source": "suggested"
            })

            await queue.put({"type": "done", "model": model_id})

        except Exception as e:
            dest.unlink(missing_ok=True)
            model_manager.set_download_state(category, model_id, "not_downloaded")
            await queue.put({"type": "error", "model": model_id, "message": str(e)})

        await queue.put({"type": "complete"})
        await queue.put(None)

    async def event_stream():
        task = asyncio.create_task(single_downloader())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── Targeted Model Purge / Delete ────────────────────────────────────────────

@router.delete("/models/{category}/{model_id}", status_code=204)
def delete_model_variant(category: str, model_id: str):
    if category not in get_all_models():
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found.")
    
    cfg = get_model_config(category, model_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Model ID '{model_id}' not found.")

    if model_manager.current_model_id() == model_id and model_manager.current_category() == category:
        model_manager.unload()
    model_manager.unload_helper_if(category, model_id)

    dest = model_path(category, model_id)
    
    try:
        model_manager.set_download_state(category, model_id, "deleting")
        if dest.exists():
            dest.unlink()
        
        model_manager.set_download_state(category, model_id, "not_downloaded")
        memory.remove_downloaded_model(model_id)
        
        if memory.get_active_model(category) == model_id:
            rec = get_recommended_model(category)
            memory.set_active_model(category, rec["id"] if rec else "")

    except Exception as e:
        if dest.exists():
            model_manager.set_download_state(category, model_id, "ready")
        raise HTTPException(status_code=500, detail=f"Failed to clear binary safely: {str(e)}")
    
class ModelSelectionRequest(BaseModel):
    model_id: str

@router.put("/models/{category}/select", status_code=200)
def update_active_model_pointer(category: str, body: ModelSelectionRequest):
    if category not in get_all_models():
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found.")
    
    cfg = get_model_config(category, body.model_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Model ID '{body.model_id}' is invalid for '{category}'.")
        
    if not model_exists(category, body.model_id):
        raise HTTPException(
            status_code=400, 
            detail="Cannot select this model. The file must be downloaded first."
        )

    memory.set_active_model(category, body.model_id)
    
    return {
        "status": "success", 
        "message": f"Active model for '{category}' successfully switched to '{body.model_id}'."
    }

# ── Search Keys ──────────────────────────────────────────────────────────────

SEARCH_PROVIDERS = ["brave", "tavily"]

class SearchKeyRequest(BaseModel):
    provider: str
    key: str

@router.get("/search-keys")
def get_search_keys():
    result = {}
    for provider in SEARCH_PROVIDERS:
        key = memory.get_search_key(provider)
        if key:
            masked = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
            result[provider] = {"has_key": True, "masked_key": masked}
        else:
            result[provider] = {"has_key": False, "masked_key": None}
    return result

@router.post("/search-keys")
def set_search_key(body: SearchKeyRequest):
    if body.provider not in SEARCH_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{body.provider}'")
    memory.set_search_key(body.provider, body.key.strip())
    return {"status": "success"}