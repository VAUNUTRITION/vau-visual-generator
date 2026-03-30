#!/usr/bin/env python3
"""
VAU Nutrition — Campaign Visual Generator
Railway service que llama a Higgsfield API para generación de imágenes.

Endpoints:
  GET  /health              — health check
  POST /upload              — upload product images para una campaña
  POST /generate            — lanzar job de generación
  GET  /status/{job_id}     — polling de status
  GET  /results/{job_id}    — URLs de imágenes cuando done
  GET  /jobs                — listar todos los jobs
"""

import asyncio
import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import requests
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_KEY = os.environ.get("HF_KEY", "")
# Higgsfield model for image generation
HF_MODEL = os.environ.get("HF_MODEL", "higgsfield/soul/v2/text-to-image")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

UPLOADS_DIR = Path("/tmp/vau-uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = Path("/tmp/vau-outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# In-memory job store
JOBS: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="VAU Campaign Visual Generator — Higgsfield", version="3.0.0")

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

ASPECT_RATIO_TO_SIZE = {
    "1:1":  {"width": 1024, "height": 1024},
    "4:5":  {"width": 1024, "height": 1280},
    "9:16": {"width": 768,  "height": 1344},
    "16:9": {"width": 1344, "height": 768},
    "3:4":  {"width": 896,  "height": 1152},
    "4:3":  {"width": 1152, "height": 896},
}


def get_hf_credentials():
    """Parse HF_KEY into api_key and api_secret."""
    if not HF_KEY:
        raise HTTPException(500, "HF_KEY no configurada. Agregarla en Railway → Variables.")
    # HF_KEY can be "api-key:api-secret" or just "api-key"
    if ":" in HF_KEY:
        parts = HF_KEY.split(":", 1)
        return parts[0], parts[1]
    return HF_KEY, ""


def upload_image_to_higgsfield(image_path: Path) -> str:
    """Upload a local image to Higgsfield and return the file URL."""
    api_key, api_secret = get_hf_credentials()

    ext = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    content_type = mime_map.get(ext, "image/jpeg")

    headers = {"Authorization": f"Bearer {api_key}"}
    if api_secret:
        headers["X-Api-Secret"] = api_secret

    with open(image_path, "rb") as f:
        response = requests.post(
            "https://api.higgsfield.ai/v1/files",
            headers=headers,
            files={"file": (image_path.name, f, content_type)},
            timeout=60,
        )

    if response.status_code == 200:
        data = response.json()
        return data.get("url", data.get("file_url", ""))
    else:
        print(f"Upload failed ({response.status_code}): {response.text}")
        return ""


def generate_image_higgsfield(
    prompt: str,
    product_image_urls: List[str],
    aspect_ratio: str = "1:1",
) -> Optional[bytes]:
    """
    Call Higgsfield API to generate an image.
    Returns image bytes or None on failure.
    """
    api_key, api_secret = get_hf_credentials()

    size = ASPECT_RATIO_TO_SIZE.get(aspect_ratio, {"width": 1024, "height": 1024})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if api_secret:
        headers["X-Api-Secret"] = api_secret

    # Build the request payload
    payload = {
        "model": HF_MODEL,
        "prompt": prompt,
        "width": size["width"],
        "height": size["height"],
        "num_images": 1,
    }

    # If we have product reference images, use image-to-image or soul mode
    if product_image_urls:
        payload["reference_image_urls"] = product_image_urls

    # Submit the generation request
    response = requests.post(
        "https://api.higgsfield.ai/v1/generations",
        headers=headers,
        json=payload,
        timeout=30,
    )

    if response.status_code not in (200, 201, 202):
        print(f"Generation submit failed ({response.status_code}): {response.text}")
        return None

    result = response.json()
    request_id = result.get("id") or result.get("request_id") or result.get("generation_id")

    if not request_id:
        # Synchronous response — image data directly in response
        image_url = extract_image_url(result)
        if image_url:
            return download_image_bytes(image_url)
        print(f"No request_id or image in response: {result}")
        return None

    # Poll for completion
    for attempt in range(120):  # up to ~10 minutes
        time.sleep(5)

        status_resp = requests.get(
            f"https://api.higgsfield.ai/v1/generations/{request_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )

        if status_resp.status_code != 200:
            continue

        status_data = status_resp.json()
        state = (
            status_data.get("status")
            or status_data.get("state")
            or ""
        ).lower()

        if state in ("completed", "done", "success", "finished"):
            image_url = extract_image_url(status_data)
            if image_url:
                return download_image_bytes(image_url)
            print(f"Completed but no image URL found: {status_data}")
            return None

        if state in ("failed", "error", "nsfw", "cancelled"):
            print(f"Generation failed with state: {state} — {status_data}")
            return None

    print(f"Timeout waiting for generation {request_id}")
    return None


def extract_image_url(data: dict) -> Optional[str]:
    """Extract image URL from various response formats."""
    # Try common response fields
    for field in ["image_url", "url", "output_url", "result_url"]:
        if data.get(field):
            return data[field]

    # Try nested results/images/outputs
    for list_field in ["images", "results", "outputs", "output"]:
        items = data.get(list_field)
        if isinstance(items, list) and items:
            item = items[0]
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                for field in ["url", "image_url", "output_url"]:
                    if item.get(field):
                        return item[field]

    return None


def download_image_bytes(url: str) -> Optional[bytes]:
    """Download image from URL and return bytes."""
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content
    except Exception as e:
        print(f"Download failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Save generated image
# ---------------------------------------------------------------------------

def save_image(image_bytes: bytes, campaign: str, shot_name: str, variant: int) -> str:
    out_dir = OUTPUTS_DIR / campaign
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{shot_name}_v{variant}.png"
    dest = out_dir / filename
    dest.write_bytes(image_bytes)
    return f"/download/{campaign}/{filename}"

# ---------------------------------------------------------------------------
# Background generation task
# ---------------------------------------------------------------------------

def run_generation(job_id: str, request: GenerateRequest):
    job = JOBS[job_id]
    job["status"] = "running"
    job["updated_at"] = time.time()

    try:
        prompts = request.prompts
        if request.shots:
            prompts = [p for p in prompts if p.shot_number in request.shots]

        job["total_shots"] = len(prompts)
        all_image_urls = []

        # Upload product images once if needed
        campaign_dir = UPLOADS_DIR / request.campaign
        uploaded_product_urls = []

        # Find all product images for this campaign
        if campaign_dir.exists():
            for img_file in sorted(campaign_dir.iterdir()):
                if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    print(f"[{job_id}] Uploading product image: {img_file.name}")
                    url = upload_image_to_higgsfield(img_file)
                    if url:
                        uploaded_product_urls.append(url)
                        print(f"[{job_id}]   ✓ Uploaded: {img_file.name}")

        for i, entry in enumerate(prompts):
            print(f"[{job_id}] Shot {entry.shot_number}: {entry.shot_name}")

            # Use product images only if this shot needs them
            ref_urls = uploaded_product_urls if entry.needs_product_images else []

            try:
                for v in range(request.num_images):
                    img_bytes = generate_image_higgsfield(
                        prompt=entry.prompt,
                        product_image_urls=ref_urls,
                        aspect_ratio=entry.aspect_ratio,
                    )

                    if img_bytes:
                        url = save_image(img_bytes, request.campaign, entry.shot_name, v + 1)
                        all_image_urls.append(url)
                        print(f"[{job_id}]   ✓ {entry.shot_name}_v{v+1}.png")
                    else:
                        print(f"[{job_id}]   ✗ {entry.shot_name}_v{v+1} — no image returned")

            except Exception as e:
                print(f"[{job_id}]   ✗ Shot {entry.shot_number} failed: {e}")

            job["completed_shots"] = i + 1
            job["image_urls"] = all_image_urls
            job["updated_at"] = time.time()

            time.sleep(1)  # small delay between shots

        job["status"] = "done"
        job["image_urls"] = all_image_urls
        job["updated_at"] = time.time()
        print(f"[{job_id}] ✓ Done — {len(all_image_urls)} image(s) generated")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["updated_at"] = time.time()
        print(f"[{job_id}] ✗ Job failed: {e}")

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
    """Upload product photos for a campaign. Call before /generate."""
    campaign_dir = UPLOADS_DIR / campaign
    campaign_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for file in files:
        if Path(file.filename).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        dest = campaign_dir / file.filename
        content = await file.read()
        dest.write_bytes(content)
        saved.append(file.filename)

    return {
        "campaign": campaign,
        "uploaded": saved,
        "total": len(saved),
        "message": f"{len(saved)} image(s) ready for generation.",
    }

@app.post("/generate")
def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Launch generation job. Returns job_id — poll /status/{job_id}."""
    if not HF_KEY:
        raise HTTPException(500, "HF_KEY not configured in Railway.")

    if not request.prompts:
        raise HTTPException(400, "No prompts in request.")

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

    return {
        "job_id": job_id,
        "status": "queued",
        "total_shots": len(prompts_to_run),
        "message": f"Job queued. Poll /status/{job_id}",
    }

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
        raise HTTPException(400, f"Job not done yet. Status: {job['status']}")
    return {
        "job_id": job_id,
        "campaign": job["campaign"],
        "product": job["product"],
        "image_urls": job["image_urls"],
        "total_images": len(job["image_urls"]),
    }

@app.get("/download/{campaign}/{filename}")
def download_image(campaign: str, filename: str):
    from fastapi.responses import FileResponse
    path = OUTPUTS_DIR / campaign / filename
    if not path.exists():
        raise HTTPException(404, "Image not found.")
    return FileResponse(path, media_type="image/png", filename=filename)

@app.get("/jobs")
def list_jobs():
    return {
        "total": len(JOBS),
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "campaign": j["campaign"],
                "completed_shots": j["completed_shots"],
                "total_shots": j["total_shots"],
            }
            for j in JOBS.values()
        ],
    }
