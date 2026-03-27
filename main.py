#!/usr/bin/env python3
"""
VAU Nutrition — Campaign Visual Generator
Railway service que llama Nano Banana 2 (gemini-3.1-flash-image-preview)
directamente via la Gemini API. Sin Fal AI.

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
from google import genai
from google.genai import types
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-3.1-flash-image-preview"

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

app = FastAPI(title="VAU Campaign Visual Generator — Gemini API", version="2.0.0")

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
    shots: Optional[List[int]] = None  # subset de shot_numbers a generar
    num_images: int = 1                # variantes por shot (1–4)
    resolution: str = "2K"            # "1K", "2K", "4K" — info only, Gemini no lo expone directamente

# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

ASPECT_RATIO_MAP = {
    "1:1":  "1:1",
    "4:5":  "4:5",
    "9:16": "9:16",
    "16:9": "16:9",
    "3:4":  "3:4",
    "4:3":  "4:3",
}

def get_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY no configurada. Agregarla en Railway → Variables.")
    return genai.Client(api_key=GEMINI_API_KEY)

def load_image_as_part(image_path: Path) -> types.Part:
    """Convierte una imagen local a un Part de Gemini (inline base64)."""
    ext = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/jpeg")
    data = base64.b64encode(image_path.read_bytes()).decode()
    return types.Part(inline_data=types.Blob(mime_type=mime, data=data))

def generate_images(
    client: genai.Client,
    prompt: str,
    product_image_paths: List[Path],
    aspect_ratio: str = "1:1",
    num_images: int = 1,
) -> List[bytes]:
    """
    Llama a Nano Banana 2 (gemini-3.1-flash-image-preview).
    Si hay imágenes de producto, las manda como referencia visual.
    Devuelve lista de bytes de imágenes PNG.
    """
    contents = []

    # Si hay imágenes del producto, agregarlas primero como referencia
    for path in product_image_paths[:14]:
        if path.exists():
            contents.append(load_image_as_part(path))

    # El prompt de texto va al final
    contents.append(types.Part(text=prompt))

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_generation_config=types.ImageGenerationConfig(
            number_of_images=min(max(num_images, 1), 4),
            aspect_ratio=ASPECT_RATIO_MAP.get(aspect_ratio, "1:1"),
        ),
    )

    results = []
    for _ in range(min(num_images, 4)):
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                results.append(base64.b64decode(part.inline_data.data))
        if len(results) >= num_images:
            break

    return results

# ---------------------------------------------------------------------------
# Guardar imagen generada y devolver URL pública
# ---------------------------------------------------------------------------

def save_image(image_bytes: bytes, campaign: str, shot_name: str, variant: int) -> str:
    """
    Guarda la imagen en /tmp/vau-outputs/{campaign}/{shot_name}_v{variant}.png
    y devuelve la URL de descarga via /download/{campaign}/{filename}
    """
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
        client = get_client()

        prompts = request.prompts
        if request.shots:
            prompts = [p for p in prompts if p.shot_number in request.shots]

        job["total_shots"] = len(prompts)
        all_image_urls = []

        for i, entry in enumerate(prompts):
            print(f"[{job_id}] Shot {entry.shot_number}: {entry.shot_name}")

            # Resolver paths de imágenes del producto
            campaign_dir = UPLOADS_DIR / request.campaign
            product_paths = []
            for fname in entry.product_image_files:
                path = campaign_dir / fname
                if path.exists():
                    product_paths.append(path)
                else:
                    print(f"[{job_id}]   ⚠ {fname} no encontrado — generando sin referencia")

            try:
                images_bytes = generate_images(
                    client=client,
                    prompt=entry.prompt,
                    product_image_paths=product_paths,
                    aspect_ratio=entry.aspect_ratio,
                    num_images=request.num_images,
                )

                for j, img_bytes in enumerate(images_bytes, start=1):
                    url = save_image(img_bytes, request.campaign, entry.shot_name, j)
                    all_image_urls.append(url)
                    print(f"[{job_id}]   ✓ {entry.shot_name}_v{j}.png")

            except Exception as e:
                print(f"[{job_id}]   ✗ Shot {entry.shot_number} falló: {e}")

            job["completed_shots"] = i + 1
            job["image_urls"] = all_image_urls
            job["updated_at"] = time.time()

            time.sleep(0.5)  # pequeño delay entre shots

        job["status"] = "done"
        job["image_urls"] = all_image_urls
        job["updated_at"] = time.time()
        print(f"[{job_id}] ✓ Done — {len(all_image_urls)} imagen(es) generadas")

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
        "model": MODEL,
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "active_jobs": len(JOBS),
    }

@app.post("/upload")
async def upload_product_images(
    campaign: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Subir fotos del producto para una campaña. Llamar antes de /generate."""
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
        "message": f"{len(saved)} imagen(es) lista(s) para generación.",
    }

@app.post("/generate")
def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Lanzar job de generación. Devuelve job_id — hacer polling en /status/{job_id}."""
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY no configurada en Railway.")

    if not request.prompts:
        raise HTTPException(400, "No hay prompts en el request.")

    prompts_to_run = request.prompts
    if request.shots:
        prompts_to_run = [p for p in prompts_to_run if p.shot_number in request.shots]
        if not prompts_to_run:
            raise HTTPException(400, f"No se encontraron shots {request.shots}.")

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
        "message": f"Job en cola. Hacer polling en /status/{job_id}",
    }

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} no encontrado.")
    return job

@app.get("/results/{job_id}")
def get_results(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} no encontrado.")
    if job["status"] != "done":
        raise HTTPException(400, f"Job no terminó aún. Status: {job['status']}")
    return {
        "job_id": job_id,
        "campaign": job["campaign"],
        "product": job["product"],
        "image_urls": job["image_urls"],
        "total_images": len(job["image_urls"]),
    }

@app.get("/download/{campaign}/{filename}")
def download_image(campaign: str, filename: str):
    """Descargar una imagen generada por filename."""
    from fastapi.responses import FileResponse
    path = OUTPUTS_DIR / campaign / filename
    if not path.exists():
        raise HTTPException(404, "Imagen no encontrada.")
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
