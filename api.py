# api.py — CLEAN VERSION

import os
import io
import json
import logging
import base64
import glob
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Conv2D
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# GradCAM utils
from gradcam_utils import (
    preprocess_pil,
    make_gradcam_heatmap,
    overlay_heatmap,
    pil_to_base64
)

# TTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
TREATMENT_JSON_PATH = os.path.join(BASE_DIR, "treatment.json")
PORT = 8000

os.makedirs(MODEL_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# FastAPI init
app = FastAPI(title="Plant Disease Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="./templates")
app.mount("/static", StaticFiles(directory="./static"), name="static")

# Treatment JSON loader
def load_treatment_json() -> Dict[str, Any]:
    """Load treatment.json from project root"""
    try:
        if os.path.exists(TREATMENT_JSON_PATH):
            with open(TREATMENT_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info("✅ Successfully loaded treatment.json")
                return data
        else:
            logger.warning("❌ treatment.json not found at: %s", TREATMENT_JSON_PATH)
            return {}
    except Exception as e:
        logger.error("❌ Failed to load treatment.json: %s", e)
        return {}

# Audio Functions
def make_audio_base64(text: str, lang="te"):
    """Convert text → base64 audio"""
    if not GTTS_AVAILABLE:
        return None
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

# Model recombine
def recombine_model_parts():
    if os.path.exists(MODEL_PATH):
        return True
    parts = sorted(glob.glob(os.path.join(MODEL_DIR, "best_model.h5.part*")))
    if not parts:
        logger.warning("No model parts found.")
        return False
    logger.info("Recombining model parts...")
    with open(MODEL_PATH, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                out.write(f.read())
    logger.info("Model recombined successfully.")
    return True

# Get Treatment Data
def get_treatment_data(disease: str, lang: str) -> Dict[str, Any]:
    """Get treatment data from treatment.json"""
    treatments = load_treatment_json()
    
    # Get disease data
    disease_data = treatments.get(disease, treatments.get("Rust", {}))
    lang_data = disease_data.get(lang, disease_data.get("en", {}))
    
    # Build treatment response
    treatment = {
        "treatment_summary": lang_data.get("treatment_summary", "Treatment information not available."),
        "step_by_step": lang_data.get("step_by_step", []),
        "pesticides": lang_data.get("pesticides", []),
        "precautions": lang_data.get("precautions", ""),
        "translations": {
            "te": disease_data.get("te", {}).get("treatment_summary", ""),
            "en": disease_data.get("en", {}).get("treatment_summary", ""),
            "hi": disease_data.get("hi", {}).get("treatment_summary", "")
        },
        "language": lang
    }
    
    return treatment

# Safe GradCAM
def safe_gradcam(img: Image.Image, model, last_conv_layer):
    """Safely generate GradCAM heatmap"""
    try:
        if model is None or last_conv_layer is None:
            return None
        x = preprocess_pil(img, (225, 225))
        heatmap = make_gradcam_heatmap(x, model, last_conv_layer)
        return heatmap
    except Exception as e:
        logger.error(f"GradCAM error: {e}")
        return None

# Load ML model
model = None
LAST_CONV = None

if TF_AVAILABLE:
    recombine_model_parts()
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            for layer in reversed(model.layers):
                if isinstance(layer, Conv2D):
                    LAST_CONV = layer.name
                    break
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            model = None
    else:
        logger.warning("❌ Model file not found")

# Routes
@app.get("/")
def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = Form("te")):
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        image_data = await file.read()
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Prediction
        if model:
            x = preprocess_pil(img, (225, 225))
            pred = model.predict(x, verbose=0)[0]
            idx = int(np.argmax(pred))
            conf = float(np.max(pred))
            label = {0: "Healthy", 1: "Powdery", 2: "Rust"}.get(idx, "Unknown")
        else:
            # Demo fallback
            label = "Rust"
            conf = 0.85

        # GradCAM
        heatmap = safe_gradcam(img, model, LAST_CONV)
        if heatmap is not None:
            overlay = overlay_heatmap(img, heatmap)
        else:
            overlay = img

        gradcam_b64 = pil_to_base64(overlay)

        # Get treatment data
        treatment = get_treatment_data(label, lang)

        # Audio
        audio_b64 = make_audio_base64(treatment["treatment_summary"], lang=lang)

        logger.info(f"✅ Prediction: {label} (confidence: {conf})")

        return {
            "disease": label,
            "confidence": conf,
            "gradcam_base64": gradcam_b64,
            "treatment": treatment,
            "treatment_audio_base64": audio_b64
        }

    except Exception as e:
        logger.exception("❌ Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/tts")
async def text_to_speech(text: str = Form(...), lang: str = Form("te")):
    """Convert text to speech audio"""
    try:
        if not GTTS_AVAILABLE:
            raise HTTPException(status_code=500, detail="TTS not available")

        valid_langs = {"te", "en", "hi"}
        if lang not in valid_langs:
            lang = "te"

        tts = gTTS(text=text, lang=lang)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.get("/download_uploaded_doc")
async def download_uploaded_doc():
    """Serve project document"""
    doc_path = os.path.join(BASE_DIR, "project_document.docx")
    if os.path.exists(doc_path):
        return FileResponse(doc_path, filename="plant_disease_project.docx")
    else:
        return JSONResponse({"error": "Document not found"}, status_code=404)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "treatment_loaded": os.path.exists(TREATMENT_JSON_PATH)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=PORT, reload=True)
