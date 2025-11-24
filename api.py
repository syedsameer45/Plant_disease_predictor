# api.py -- FastAPI with split model files from GitHub
import os
import io
import json
import logging
import base64
import urllib.request
from typing import Optional


# --- Base paths & environment ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_files")
MODEL_DIR = os.path.join(BASE_DIR, "models")
GRADCAM_OVERLAY_PATH = os.path.join(BASE_DIR, "gradcam_overlay.jpg")
TREATMENT_GUIDE_PATH = os.path.join(BASE_DIR, "treatment_guide.json")
PORT = int(os.environ.get("PORT", 8000))

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")

# create dirs if missing
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# --- Try imports that may be heavy or optional ---
try:
    import numpy as np
    from PIL import Image
except Exception as e:
    logger.exception("Required libraries missing (numpy/pillow): %s", e)
    raise

# Import TensorFlow lazily
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import load_model  # type: ignore
    TF_AVAILABLE = True
    logger.info("TensorFlow import succeeded.")
except Exception as e:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available: %s", e)

# gTTS & googletrans are optional
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception as e:
    GTTS_AVAILABLE = False
    logger.warning("gTTS not available: %s", e)

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except Exception as e:
    GOOGLETRANS_AVAILABLE = False
    logger.warning("googletrans not available: %s", e)

# FastAPI & related
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Plant Disease Detector with Grad-CAM + TTS")

# mount static files and templates
templates_dir = os.path.join(BASE_DIR, "templates")
static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = Jinja2Templates(directory=BASE_DIR)
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ---------- Model recombining and loading ----------
model = None

def recombine_model_if_needed():
    """Recombine split model files if they exist in the models/ folder"""
    import glob
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        logger.info("Model already exists at %s", MODEL_PATH)
        return True
    
    # Look for split parts - handles both naming patterns
    # Pattern 1: best_model.h5.part1, best_model.h5.part2
    part_pattern_1 = os.path.join(MODEL_DIR, "best_model.h5.part*")
    # Pattern 2: best_model (1).h5.part1, best_model (1).h5.part2
    part_pattern_2 = os.path.join(MODEL_DIR, "best_model*.h5.part*")
    
    parts = sorted(glob.glob(part_pattern_1))
    if not parts:
        parts = sorted(glob.glob(part_pattern_2))
    
    if not parts or len(parts) < 2:
        logger.warning("No split model parts found in %s", MODEL_DIR)
        return False
    
    logger.info("Found %d model parts, recombining...", len(parts))
    
    try:
        with open(MODEL_PATH, 'wb') as outfile:
            for part_file in parts:
                logger.info("  Adding %s...", os.path.basename(part_file))
                with open(part_file, 'rb') as infile:
                    outfile.write(infile.read())
        
        file_size = os.path.getsize(MODEL_PATH)
        logger.info("Model recombined successfully: %s (%d bytes)", MODEL_PATH, file_size)
        return True
    except Exception as e:
        logger.exception("Failed to recombine model: %s", e)
        return False


if TF_AVAILABLE:
    # Recombine split model parts if needed
    recombine_model_if_needed()
    
    if os.path.exists(MODEL_PATH):
        try:
            logger.info("Loading model from %s", MODEL_PATH)
            # load_model can be heavy; catch exceptions and continue in demo mode if fails
            model = load_model(MODEL_PATH)
            try:
                model(np.zeros((1, 225, 225, 3), dtype=np.float32))
                logger.info("Model loaded and warmed up.")
            except Exception:
                logger.info("Model loaded (build deferred).")
        except Exception as e:
            logger.exception("Failed to load model: %s", e)
            model = None
    else:
        logger.warning("Model file not found at %s; continuing in demo mode.", MODEL_PATH)
else:
    logger.warning("TensorFlow not available; running in demo mode (no model).")


# ---------- Helper functions ----------
def preprocess_image_bytes(contents: bytes, target_size=(225, 225)):
    pil = Image.open(io.BytesIO(contents)).convert("RGB")
    pil = pil.resize(target_size)
    arr = np.asarray(pil).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def make_dummy_gradcam_base64():
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4////GQAH/wN/9yD6AAAAAElFTkSuQmCC"


def safe_read_json(path: str):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning("Failed to read JSON at %s: %s", path, e)
        return {}


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception:
        return HTMLResponse("<h2>Plant Disease Detector API</h2><p>Use /health, /predict, /tts, /translate</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": bool(model), "tf_available": TF_AVAILABLE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse({"error": "Empty file"}, status_code=400)
        arr = preprocess_image_bytes(contents, target_size=(225, 225))
    except Exception as e:
        logger.exception("Invalid image upload")
        return JSONResponse({"error": "Invalid image upload", "detail": str(e)}, status_code=400)

    response = {
        "disease": "Unknown",
        "confidence": "0.0",
        "gradcam_base64": make_dummy_gradcam_base64(),
        "treatment": {"summary": "No treatment available.", "steps": []},
        "last_conv": ""
    }

    if model is not None and TF_AVAILABLE:
        try:
            preds = model.predict(arr)
            idx = int(np.argmax(preds[0]))
            conf = float(np.max(preds[0]))
            label_map = {0: "Healthy", 1: "Powdery", 2: "Rust"}
            label = label_map.get(idx, "Unknown")
            response["disease"] = label
            response["confidence"] = f"{conf:.6f}"
            response["last_conv"] = "conv2d_*"

            if os.path.exists(GRADCAM_OVERLAY_PATH):
                try:
                    with open(GRADCAM_OVERLAY_PATH, "rb") as f:
                        response["gradcam_base64"] = base64.b64encode(f.read()).decode("utf-8")
                except Exception:
                    response["gradcam_base64"] = make_dummy_gradcam_base64()
            else:
                response["gradcam_base64"] = make_dummy_gradcam_base64()

            guides = safe_read_json(TREATMENT_GUIDE_PATH)
            if guides:
                guide = guides.get(label)
                if guide:
                    response["treatment"] = guide
        except Exception as e:
            logger.exception("Prediction failed")
            return JSONResponse({"error": "Prediction failed", "detail": str(e)}, status_code=500)
    else:
        response["disease"] = "Demo (model not loaded)"
        response["confidence"] = "0.0"
        if os.path.exists(GRADCAM_OVERLAY_PATH):
            try:
                with open(GRADCAM_OVERLAY_PATH, "rb") as f:
                    response["gradcam_base64"] = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                response["gradcam_base64"] = make_dummy_gradcam_base64()

    return JSONResponse(response)


@app.post("/tts")
async def tts_endpoint(text: str = Form(...), lang: str = Form("en")):
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    if not GTTS_AVAILABLE:
        return JSONResponse({"error": "gTTS not installed on server"}, status_code=500)
    try:
        tts = gTTS(text=text, lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg")
    except Exception as e:
        logger.exception("TTS generation failed")
        return JSONResponse({"error": "TTS generation failed", "detail": str(e)}, status_code=500)


@app.post("/translate")
async def translate_endpoint(text: str = Form(...), target: str = Form("te")):
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    if not GOOGLETRANS_AVAILABLE:
        return JSONResponse({"error": "googletrans not installed on server"}, status_code=500)
    try:
        translator = Translator()
        result = translator.translate(text, dest=target)
        translated = result.text
        return JSONResponse({"translated": translated})
    except Exception as e:
        logger.exception("Translation failed")
        return JSONResponse({"error": "Translation failed", "detail": str(e)}, status_code=500)


@app.get("/download_uploaded_doc")
async def download_uploaded_doc():
    doc_path = os.path.join(UPLOAD_DIR, "project_document.docx")
    if not os.path.exists(doc_path):
        return JSONResponse({"error": "Uploaded doc not found"}, status_code=404)
    return FileResponse(doc_path,
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        filename="project_document.docx")


# If run directly (useful for local dev)
if __name__ == "__main__":
    import uvicorn
    reload_flag = os.environ.get("RELOAD", "false").lower() in ("1", "true", "yes")
    uvicorn.run("api:app", host="0.0.0.0", port=PORT, reload=reload_flag)
