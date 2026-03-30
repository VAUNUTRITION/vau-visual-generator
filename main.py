#!/usr/bin/env python3
"""
VAU Nutrition — Campaign Visual Generator
Railway service usando Higgsfield SDK oficial.

Endpoints:
  GET  /health              — health check
  POST /upload              — upload product images para una campaña
  POST /generate            — lanzar job de generación
  GET  /status/{job_id}     — polling de status
  GET  /results/{job_id}    — URLs de imágenes cuando done
  GET  /jobs                — listar todos los jobs
"""

import base64
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import higgsfield_client as hf
from higgsfield_client import Completed, Failed, NSFW, Cancelled
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_KEY = os.environ.get("HF_KEY", "")
HF_MODEL = os.environ.get("HF_MODEL", "bytedance/seedream/v4/text-to-image")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

UPLOADS_DIR = Path("/tmp/vau-uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = Path("/tmp/vau-outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

JOBS: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="VAU Campaign Visual Generator — Higgsfield", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PromptEntry(BaseModel):
    shot_number: int
    shot_name: str
    prompt: str
    aspect_ratio: str = "1:1"
    needs_product_images: bool = True
    product_image_files: List[str] = []
    notes: Optional[str] = None

class GenerateRequest(BaseModel):
    campaign: str
    brand: str = "VAU Nutrition"
    product: str
    style_reference: Optional[str] = None
    generated_at: Optional[str] = None
    prompts: List[PromptEntry]
    shots: Optional[List[int]] = None
    num_images: int = 1
    resolution: str = "2K"

# ---------------------------------------------------------------------------
# Higgsfield helpers
# ---------------------------------------------------------------------------

ASPECT_TO_DIMS = {
    "1:1":  (1024, 1024),
    "4:5":  (1024, 1280),
    "9:16": (768,  1344),
    "16:9": (1344, 768),
    "3:4":  (896,  1152),
    "4:3":  (1152, 896),
}


def get_hf_client() -> hf.SyncClient:
    if not HF_KEY:
        raise HTTPException(500, "HF_KEY no configurada en Railway.")
    return hf.SyncClient(api_key=HF_KEY)


def upload_ref_images(client: hf.SyncClient, image_paths: List[Path]) -> List[str]:
    """Upload product reference images to Higgsfield and return URLs."""
    urls = []
    for path in image_paths[:4]:  # max 4 reference images
        if not path.exists():
            continue
        try:
            url = client.upload_file(str(path))
            urls.append(url)
            print(f"  ↑ Uploaded ref: {path.name} → {url[:60]}...")
        except Exception as e:
            print(f"  ✗ Upload failed {path.name}: {e}")
    return urls


def generate_one_image(
    client: hf.SyncClient,
    prompt: str,
    aspect_ratio: str = "1:1",
    ref_urls: List[str] = None,
) -> Optional[bytes]:
    """Submit one image generation request and wait for result."""
    w, h = ASPECT_TO_DIMS.get(aspect_ratio, (1024, 1024))

    arguments = {
        "prompt": prompt,
        "width": w,
        "height": h,
    }

    # Add reference images if available
    if ref_urls:
        arguments["image_urls"] = ref_urls[:2]

    try:
        ctrl = client.submit(HF_MODEL, arguments)
        print(f"  → Submitted job {ctrl.request_id}")

        # Poll until done (max ~5 min)
        for status in ctrl.poll_request_status(delay=3.0):
            print(f"  ⟳ {ctrl.request_id}: {type(status).__name__}")
            if isinstance(status, (Completed, Failed, NSFW, Cancelled)):
                break

        if not isinstance(status, Completed):
            print(f"  ✗ Job ended with: {type(status).__name__}")
            return None

        # Get result
        result = ctrl.get()
        image_url = (
            result.get("image_url")
            or result.get("url")
            or (result.get("images") or [None])[0]
            or (result.get("outputs") or [None])[0]
        )

        if not image_url:
            print(f"  ✗ No image URL in result: {list(result.keys())}")
            return None

        # Download the image
        import requests as req
        resp = req.get(image_url, timeout=60)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content

        print(f"  ✗ Download failed: {resp.status_code}")
        return None

    except Exception as e:
        print(f"  ✗ Generation error: {e}")
        return None


# ---------------------------------------------------------------------------
# Save image
# ---------------------------------------------------------------------------

def save_image(image_bytes: bytes, campaign: str, shot_name: str, variant: int) -> str:
    out_dir = OUTPUTS_DIR / campaign
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{shot_name}_v{variant}.png"
    (out_dir / filename).write_bytes(image_bytes)
    return f"/download/{campaign}/{filename}"


# ---------------------------------------------------------------------------
# Background job
# ---------------------------------------------------------------------------

def run_generation(job_id: str, request: GenerateRequest):
    job = JOBS[job_id]
    job["status"] = "running"
    job["updated_at"] = time.time()

    try:
        client = get_hf_client()

        prompts = request.prompts
        if request.shots:
            prompts = [p for p in prompts if p.shot_number in request.shots]

        job["total_shots"] = len(prompts)
        all_urls = []

        # Upload product images once
        campaign_dir = UPLOADS_DIR / request.campaign
        product_ref_urls = []
        if campaign_dir.exists():
            imgs = sorted([f for f in campaign_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
            if imgs:
                print(f"[{job_id}] Uploading {len(imgs)} product image(s)...")
                product_ref_urls = upload_ref_images(client, imgs)
                print(f"[{job_id}] {len(product_ref_urls)} ref image(s) ready")

        for i, entry in enumerate(prompts):
            print(f"[{job_id}] Shot {entry.shot_number}: {entry.shot_name}")

            ref_urls = product_ref_urls if entry.needs_product_images else []

            for v in range(request.num_images):
                img_bytes = generate_one_image(
                    client=client,
                    prompt=entry.prompt,
                    aspect_ratio=entry.aspect_ratio,
                    ref_urls=ref_urls,
                )
                if img_bytes:
                    url = save_image(img_bytes, request.campaign, entry.shot_name, v + 1)
                    all_urls.append(url)
                    print(f"[{job_id}]   ✓ {entry.shot_name}_v{v+1}.png saved")
                else:
                    print(f"[{job_id}]   ✗ {entry.shot_name}_v{v+1} — no image")

            job["completed_shots"] = i + 1
            job["image_urls"] = all_urls
            job["updated_at"] = time.time()
            time.sleep(1)

        job["status"] = "done"
        job["image_urls"] = all_urls
        job["updated_at"] = time.time()
        print(f"[{job_id}] ✓ Done — {len(all_urls)} image(s)")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["updated_at"] = time.time()
        print(f"[{job_id}] ✗ Failed: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "VAU Campaign Visual Generator",
        "engine": "Higgsfield",
        "model": HF_MODEL,
        "hf_key_configured": bool(HF_KEY),
        "active_jobs": len(JOBS),
    }

@app.post("/upload")
async def upload_product_images(
    campaign: str = Form(...),
    files: List[UploadFile] = File(...),
):
    campaign_dir = UPLOADS_DIR / campaign
    campaign_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        if Path(file.filename).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        dest = campaign_dir / file.filename
        dest.write_bytes(await file.read())
        saved.append(file.filename)
    return {"campaign": campaign, "uploaded": saved, "total": len(saved)}

@app.post("/generate")
def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    if not HF_KEY:
        raise HTTPException(500, "HF_KEY not configured.")
    if not request.prompts:
        raise HTTPException(400, "No prompts.")

    prompts_to_run = request.prompts
    if request.shots:
        prompts_to_run = [p for p in prompts_to_run if p.shot_number in request.shots]
        if not prompts_to_run:
            raise HTTPException(400, f"Shots {request.shots} not found.")

    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "campaign": request.campaign,
        "product": request.product,
        "total_shots": len(prompts_to_run),
        "completed_shots": 0,
        "image_urls": [],
        "error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    background_tasks.add_task(run_generation, job_id, request)
    return {"job_id": job_id, "status": "queued", "total_shots": len(prompts_to_run), "message": f"Poll /status/{job_id}"}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found.")
    return job

@app.get("/results/{job_id}")
def get_results(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found.")
    if job["status"] != "done":
        raise HTTPException(400, f"Not done yet. Status: {job['status']}")
    return {"job_id": job_id, "campaign": job["campaign"], "image_urls": job["image_urls"], "total_images": len(job["image_urls"])}

@app.get("/download/{campaign}/{filename}")
def download_image(campaign: str, filename: str):
    from fastapi.responses import FileResponse
    path = OUTPUTS_DIR / campaign / filename
    if not path.exists():
        raise HTTPException(404, "Image not found.")
    return FileResponse(path, media_type="image/png", filename=filename)

@app.get("/jobs")
def list_jobs():
    return {"total": len(JOBS), "jobs": [{"job_id": j["job_id"], "status": j["status"], "campaign": j["campaign"], "completed_shots": j["completed_shots"], "total_shots": j["total_shots"]} for j in JOBS.values()]}
