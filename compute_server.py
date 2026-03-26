"""
Segmentum — Compute Server v2
==============================
Changes:
  - Masks exported as uint16 TIF (not PNG)
  - Diameter override slider support
  - Model override (backend, model name)
  - Sensitivity slider → flow_threshold mapping
  - /train endpoint: upload up to 10 image+mask pairs → fine-tune Cellpose
  - PlantSeg added to available backends
"""

from __future__ import annotations

import base64
import io
import json
import os
import traceback
import uuid
from pathlib import Path
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from image_detector import detect_image
from parameter_suggester import suggest_parameters
from backend_wrappers import BackendDispatcher

app = FastAPI(
    title="Segmentum Compute API v2",
    description="GPU-powered bioimage segmentation",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dispatcher = BackendDispatcher()

# Training data stored locally
TRAINING_DIR = Path("training_data")
TRAINING_DIR.mkdir(exist_ok=True)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ImageProfileResponse(BaseModel):
    axes: str
    bit_depth: int
    is_3d: bool
    has_time: bool
    n_channels: int
    n_z: int
    n_timepoints: int
    height: int
    width: int
    object_type: str
    object_density: str
    estimated_diameter_px: float
    estimated_object_count: int
    global_snr: float
    has_clipping: bool
    warnings: list[str]
    summary: str

class PlanResponse(BaseModel):
    backend: str
    model_name: str
    diameter: Optional[float]
    flow_threshold: float
    cellprob_threshold: float
    use_3d: bool
    primary_channel: int
    confidence: str
    reasoning: list[str]
    alternative_backends: list[str]
    preprocessing: list[str]
    analysis_templates: list[dict]

class SegmentationResponse(BaseModel):
    job_id: str
    n_objects: int
    backend_used: str
    model_used: str
    runtime_seconds: float
    warnings: list[str]
    masks_tif_b64: str    # uint16 TIF — proper label format
    masks_png_b64: str    # normalised PNG for display only
    overlay_b64: str
    measurements: list[dict]
    plot_data: dict

class HealthResponse(BaseModel):
    status: str
    available_backends: list[str]
    gpu_available: bool
    version: str
    training_pairs: int   # how many GT pairs have been uploaded

class TrainResponse(BaseModel):
    status: str
    message: str
    n_pairs: int
    model_path: Optional[str]


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False

    n_pairs = len(list(TRAINING_DIR.glob("*_image.*")))

    return HealthResponse(
        status="ok",
        available_backends=dispatcher.available_backends(),
        gpu_available=gpu,
        version="0.2.0",
        training_pairs=n_pairs,
    )


# ── Serve analyse page directly (bypasses Netlify CSP issues) ─────────────────

@app.get("/analyse", response_class=HTMLResponse)
async def serve_analyse():
    analyse_path = Path(__file__).parent / "analyse.html"
    if analyse_path.exists():
        return HTMLResponse(content=analyse_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>analyse.html not found</h1>", status_code=404)


# ── Detect ────────────────────────────────────────────────────────────────────

@app.post("/detect", response_model=ImageProfileResponse)
async def detect_endpoint(
    file: UploadFile = File(...),
    channel_names: Optional[str] = Form(None),
):
    try:
        contents = await file.read()
        arr = _load_array(contents, file.filename)
        ch_names = channel_names.split(",") if channel_names else None
        profile = detect_image(arr, channel_names=ch_names)
        return ImageProfileResponse(
            axes=profile.axes, bit_depth=profile.bit_depth,
            is_3d=profile.is_3d, has_time=profile.has_time,
            n_channels=profile.n_channels, n_z=profile.n_z,
            n_timepoints=profile.n_timepoints,
            height=profile.height, width=profile.width,
            object_type=profile.object_type,
            object_density=profile.object_density,
            estimated_diameter_px=profile.estimated_diameter_px,
            estimated_object_count=profile.estimated_object_count,
            global_snr=profile.global_snr,
            has_clipping=profile.has_clipping,
            warnings=profile.warnings, summary=profile.summary,
        )
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {e}\n{traceback.format_exc()}")


# ── Suggest ───────────────────────────────────────────────────────────────────

@app.post("/suggest", response_model=PlanResponse)
async def suggest_endpoint(
    file: UploadFile = File(...),
    channel_names: Optional[str] = Form(None),
    prefer_gpu: bool = Form(True),
):
    try:
        contents = await file.read()
        arr = _load_array(contents, file.filename)
        ch_names = channel_names.split(",") if channel_names else None
        profile = detect_image(arr, channel_names=ch_names)
        plan = suggest_parameters(profile, prefer_gpu=prefer_gpu)
        p = plan.backend_params
        return PlanResponse(
            backend=p.backend, model_name=p.model_name,
            diameter=p.diameter,
            flow_threshold=p.flow_threshold,
            cellprob_threshold=p.cellprob_threshold,
            use_3d=p.use_3d,
            primary_channel=plan.primary_channel,
            confidence=plan.confidence,
            reasoning=plan.reasoning,
            alternative_backends=plan.alternative_backends,
            preprocessing=plan.preprocessing,
            analysis_templates=[
                {"name": t.name, "label": t.label, "enabled": t.enabled}
                for t in plan.analysis_templates
            ],
        )
    except Exception as e:
        raise HTTPException(500, f"Suggestion failed: {e}")


# ── Segment (main endpoint) ───────────────────────────────────────────────────

@app.post("/segment", response_model=SegmentationResponse)
async def segment_endpoint(
    file: UploadFile = File(...),
    channel_names: Optional[str] = Form(None),
    prefer_gpu: bool = Form(True),
    # Override parameters from UI
    override_backend: Optional[str]  = Form(None),   # cellpose|stardist|plantseg|threshold
    override_model:   Optional[str]  = Form(None),   # cyto3|nuclei|2D_versatile_fluo|etc
    override_diameter: Optional[float] = Form(None), # pixels
    override_sensitivity: Optional[float] = Form(None),  # 0.0–1.0 slider → flow_threshold
    override_object_type: Optional[str] = Form(None), # cells|nuclei|organoids|cysts|plant
    quality_mode: bool = Form(False),  # True = use foundation models (slower, better)
):
    job_id = str(uuid.uuid4())[:8]
    try:
        contents = await file.read()
        arr = _load_array(contents, file.filename)
        ch_names = channel_names.split(",") if channel_names else None

        profile = detect_image(arr, channel_names=ch_names)

        # Apply object type override before suggestion
        if override_object_type:
            profile.object_type = override_object_type

        plan = suggest_parameters(profile, prefer_gpu=prefer_gpu)

        # Apply parameter overrides from UI sliders/dropdowns
        if override_backend:
            plan.backend_params.backend = override_backend
        if override_model:
            plan.backend_params.model_name = override_model
        if override_diameter is not None:
            plan.backend_params.diameter = float(override_diameter)
        if override_sensitivity is not None:
            plan.backend_params.flow_threshold   = 0.8 - (float(override_sensitivity) * 0.7)
            plan.backend_params.cellprob_threshold = -3.0 + (float(override_sensitivity) * 3.0)

        # Pass object_type and quality_mode to backend
        plan.backend_params.extra["object_type"] = profile.object_type
        plan.backend_params.extra["quality_mode"] = quality_mode

        result   = dispatcher.segment(arr, plan)
        measures = _quantify(arr, result.masks, profile)
        plot_data = _build_plot_data(measures, profile)

        # TIF masks (uint16, proper label format) — main download
        masks_tif_b64 = base64.b64encode(result.masks_as_tif_bytes()).decode()
        # PNG masks (display only)
        masks_png_b64 = _masks_to_b64_png(result.masks)
        overlay_b64   = _make_overlay_b64(arr, result.masks, profile.primary_channel)

        return SegmentationResponse(
            job_id=job_id,
            n_objects=result.n_objects,
            backend_used=result.backend_used,
            model_used=result.model_used,
            runtime_seconds=result.runtime_seconds,
            warnings=result.warnings + profile.warnings,
            masks_tif_b64=masks_tif_b64,
            masks_png_b64=masks_png_b64,
            overlay_b64=overlay_b64,
            measurements=measures,
            plot_data=plot_data,
        )

    except Exception as e:
        raise HTTPException(500,
            f"[job {job_id}] Segmentation failed: {e}\n{traceback.format_exc()}")


# ── Ground truth upload + fine-tuning ─────────────────────────────────────────

@app.post("/upload_ground_truth")
async def upload_ground_truth(
    image: UploadFile = File(...),
    mask:  UploadFile = File(...),
):
    """
    Upload one image + corresponding ground truth mask.
    Stores up to 10 pairs. Used for fine-tuning.
    """
    existing = list(TRAINING_DIR.glob("*_image.*"))
    if len(existing) >= 10:
        raise HTTPException(400,
            "Maximum 10 training pairs reached. "
            "Call /train to fine-tune, then /clear_training_data to reset.")

    pair_id = str(uuid.uuid4())[:6]
    img_bytes  = await image.read()
    mask_bytes = await mask.read()

    img_suffix  = Path(image.filename or "img.tif").suffix or ".tif"
    mask_suffix = Path(mask.filename or "mask.tif").suffix or ".tif"

    img_path  = TRAINING_DIR / f"{pair_id}_image{img_suffix}"
    mask_path = TRAINING_DIR / f"{pair_id}_mask{mask_suffix}"

    img_path.write_bytes(img_bytes)
    mask_path.write_bytes(mask_bytes)

    n_pairs = len(list(TRAINING_DIR.glob("*_image.*")))
    return {
        "status": "ok",
        "pair_id": pair_id,
        "n_pairs": n_pairs,
        "message": f"Pair {n_pairs}/10 stored. "
                   f"{'Ready to train!' if n_pairs >= 3 else f'Upload {3 - n_pairs} more to enable training.'}"
    }


@app.get("/training_status")
def training_status():
    """How many GT pairs are stored, and whether training is possible."""
    pairs = list(TRAINING_DIR.glob("*_image.*"))
    return {
        "n_pairs": len(pairs),
        "ready_to_train": len(pairs) >= 3,
        "max_pairs": 10,
        "pairs": [p.stem.replace("_image", "") for p in pairs],
    }


@app.post("/train", response_model=TrainResponse)
async def train_endpoint(
    model_name: str = Form("cyto3"),
    n_epochs: int   = Form(100),
):
    """
    Fine-tune Cellpose on uploaded ground truth pairs.
    Minimum 3 pairs, maximum 10.
    Returns path to fine-tuned model.
    """
    pairs = list(TRAINING_DIR.glob("*_image.*"))
    if len(pairs) < 3:
        raise HTTPException(400,
            f"Need at least 3 training pairs, only have {len(pairs)}. "
            "Upload more via /upload_ground_truth.")

    try:
        import tifffile
        from cellpose import models, train

        # Load all pairs
        images, masks = [], []
        for img_path in sorted(pairs):
            pair_id = img_path.stem.replace("_image", "")
            mask_candidates = list(TRAINING_DIR.glob(f"{pair_id}_mask.*"))
            if not mask_candidates:
                continue

            img_arr  = _load_array(img_path.read_bytes(), img_path.name)
            mask_arr = _load_array(mask_candidates[0].read_bytes(), mask_candidates[0].name)

            # Ensure 2D grayscale first
            if img_arr.ndim == 3 and img_arr.shape[2] in (3, 4):
                img_arr = np.mean(img_arr[:, :, :3], axis=2).astype(np.float32)
            elif img_arr.ndim == 3 and img_arr.shape[0] <= 8:
                img_arr = np.max(img_arr, axis=0).astype(np.float32)
            elif img_arr.ndim == 3:
                img_arr = np.max(img_arr, axis=0).astype(np.float32)

            # Normalize to 0-1
            mn, mx = img_arr.min(), img_arr.max()
            if mx > mn:
                img_arr = (img_arr - mn) / (mx - mn)

            # Cellpose train_seg needs (C, H, W) — add channel axis
            img_arr = img_arr[np.newaxis].astype(np.float32)  # → (1, H, W)

            # Flatten mask to 2D
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0] if mask_arr.shape[2] <= 4 else mask_arr[0]
            mask_arr = mask_arr.astype(np.uint16)

            images.append(img_arr)
            masks.append(mask_arr)

        # Fine-tune
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        output_model_name = f"segmentum_custom_{len(images)}pairs"

        model = models.CellposeModel(pretrained_model=model_name, gpu=True)
        train.train_seg(
            model.net,
            train_data=images,
            train_labels=masks,
            channels=[1, 0],
            save_path=str(model_dir),
            save_every=50,
            n_epochs=n_epochs,
            model_name=output_model_name,
            min_train_masks=1,
        )

        model_path = str(model_dir / f"models/{output_model_name}")
        return TrainResponse(
            status="ok",
            message=f"Fine-tuned on {len(images)} pairs for {n_epochs} epochs.",
            n_pairs=len(images),
            model_path=model_path,
        )

    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}\n{traceback.format_exc()}")


@app.delete("/clear_training_data")
def clear_training_data():
    """Remove all stored training pairs."""
    removed = 0
    for f in TRAINING_DIR.iterdir():
        f.unlink()
        removed += 1
    return {"status": "ok", "removed": removed}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_array(contents: bytes, filename: str) -> np.ndarray:
    fname = (filename or "").lower()
    if fname.endswith((".tif", ".tiff")):
        import tifffile
        return tifffile.imread(io.BytesIO(contents))
    else:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(contents))
        return np.array(img)


def _get_2d_plane(image: np.ndarray, channel: int) -> np.ndarray:
    img = image.astype(float)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[0] <= 8:
            return img[min(channel, img.shape[0]-1)]
        if img.shape[2] in (3, 4):
            return np.mean(img[:, :, :3], axis=2)
        return np.max(img, axis=0)
    if img.ndim == 4:
        return np.max(img[min(channel, img.shape[0]-1)], axis=0)
    return img


def _quantify(image: np.ndarray, masks: np.ndarray, profile) -> list[dict]:
    try:
        from skimage import measure
        intensity_plane = _get_2d_plane(image, profile.primary_channel)
        # Ensure masks and intensity plane are the same shape
        if masks.shape != intensity_plane.shape[-2:]:
            intensity_plane = intensity_plane[:masks.shape[0], :masks.shape[1]]
        props = measure.regionprops(masks.astype(int), intensity_image=intensity_plane)
        measurements = []
        for p in props:
            circ = (4 * np.pi * p.area) / (p.perimeter ** 2) if p.perimeter > 0 else 0.0
            measurements.append({
                "label":          int(p.label),
                "area_px":        float(p.area),
                "perimeter_px":   float(p.perimeter),
                "eccentricity":   float(p.eccentricity),
                "circularity":    float(circ),
                "solidity":       float(p.solidity),
                "major_axis_px":  float(p.major_axis_length),
                "minor_axis_px":  float(p.minor_axis_length),
                "centroid_y":     float(p.centroid[0]),
                "centroid_x":     float(p.centroid[1]),
                "mean_intensity": float(p.mean_intensity),
                "max_intensity":  float(p.max_intensity),
                "min_intensity":  float(p.min_intensity),
            })
        return measurements
    except Exception as e:
        return [{"error": str(e)}]


def _build_plot_data(measurements: list[dict], profile) -> dict:
    if not measurements or "error" in measurements[0]:
        return {}
    areas  = [m["area_px"]      for m in measurements]
    eccen  = [m["eccentricity"] for m in measurements]
    circs  = [m["circularity"]  for m in measurements]
    cx     = [m["centroid_x"]   for m in measurements]
    cy     = [m["centroid_y"]   for m in measurements]
    return {
        "area_histogram": {
            "values": areas,
            "xlabel": "Area (px²)", "ylabel": "Count",
            "title": f"Area Distribution (n={len(areas)})",
        },
        "morphology_scatter": {
            "x": eccen, "y": areas, "color": circs,
            "xlabel": "Eccentricity", "ylabel": "Area (px²)",
            "clabel": "Circularity", "title": "Morphology Scatter",
        },
        "spatial_map": {
            "x": cx, "y": cy, "color": areas,
            "xlabel": "X (px)", "ylabel": "Y (px)",
            "title": "Spatial Map",
            "image_width": profile.width, "image_height": profile.height,
        },
        "summary_stats": {
            "n_objects":         len(areas),
            "mean_area":         float(np.mean(areas)),
            "median_area":       float(np.median(areas)),
            "std_area":          float(np.std(areas)),
            "mean_eccentricity": float(np.mean(eccen)),
            "mean_circularity":  float(np.mean(circs)),
        },
    }


def _masks_to_b64_png(masks: np.ndarray) -> str:
    """Normalised 8-bit PNG for display only."""
    from PIL import Image as PILImage
    mx = masks.max()
    arr8 = (masks.astype(np.float32) / mx * 255).astype(np.uint8) if mx > 0 else masks.astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_overlay_b64(image: np.ndarray, masks: np.ndarray, primary_channel: int) -> str:
    from PIL import Image as PILImage
    from skimage import exposure

    img_2d   = _get_2d_plane(image, primary_channel)
    img_norm = exposure.rescale_intensity(img_2d.astype(float), out_range=(0, 1))
    img_rgb  = np.stack([img_norm] * 3, axis=-1)
    overlay  = img_rgb.copy()

    rng = np.random.default_rng(42)
    n_labels = int(masks.max())
    if n_labels > 0:
        colours = rng.random((n_labels + 1, 3)) * 0.7 + 0.3
        for label_id in range(1, n_labels + 1):
            m = masks == label_id
            if m.any():
                overlay[m] = 0.35 * img_rgb[m] + 0.65 * colours[label_id]

    buf = io.BytesIO()
    PILImage.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n🔬 Segmentum Compute Server v2")
    print("   Docs:   http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health\n")
    uvicorn.run("compute_server:app", host="0.0.0.0", port=8000, reload=True)
