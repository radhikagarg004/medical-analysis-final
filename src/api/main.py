from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from src.api.database import SessionLocal, PatientRecord

# 🔥 Radhika's real AI modules
try:
    from src.predict import predict_image
    from src.uncertainty import estimate_uncertainty
    from src.image_quality import check_image_quality
    from src.gradcam import generate_gradcam
    from src.similarity import find_similar_images
    AI_READY = True
    print("✅ Real AI Model Loaded Successfully!")
except Exception as e:
    print(f"⚠️ AI not loaded: {e}")
    AI_READY = False

app = FastAPI(title="MediScan AI API", version="4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Grad-CAM output dir
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Single source of truth for port (frontend expects 8000)
BASE_URL = os.getenv("MEDVISION_BASE_URL", "http://localhost:8000")
UNCERTAINTY_THRESHOLD = 2.0   # from project summary

@app.get("/health")
def health_check():
    return {"status": "ok", "ai_mode": "REAL" if AI_READY else "MOCK"}

@app.post("/check-quality")
async def check_quality_only(file: UploadFile = File(...)):
    """Standalone image quality check — no diagnosis, no DB save."""
    temp_path = None
    try:
        suffix = Path(file.filename).suffix.lower() or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(await file.read())
            temp_path = temp.name

        if not AI_READY:
            return {"status": "Unknown", "reason": "AI model not loaded"}

        result = check_image_quality(temp_path)
        return {
            "status": result.get("quality_status", "Unknown"),
            "reason": result.get("reason", ""),
            "metrics": result
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

@app.post("/predict/diagnosis")
async def predict_diagnosis(
    file: UploadFile = File(...),
    patient_name: str = Form(...),
    age: int = Form(...),
    symptoms: str = Form("")
):
    temp_path = None
    try:
        suffix = Path(file.filename).suffix.lower() or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # Baseline response — matches what dashboard.html reads
        final_result = {
            "diagnosis": "PNEUMONIA (Mock)",
            "confidence": 93.4,
            "risk": "HIGH",
            "opacity": 18.5,
            "recommendation": "Please consult a doctor.",
            "probabilities": {"COVID19": 5, "NORMAL": 10, "PNEUMONIA": 80, "TUBERCULOSIS": 5},
            "uncertainty": 2.5,
            "needs_review": False,
            "quality": "Acceptable",
            "gradcam_url": None,
            "similar": [],           # frontend reads data.similar
        }

        if AI_READY:
            # ---- 1. Quality ----
            try:
                quality = check_image_quality(temp_path)
                final_result["quality"] = quality.get("quality_status", "Unknown")
            except Exception as e:
                print(f"Quality error: {e}")

            # ---- 2. Prediction ----
            try:
                prediction = predict_image(temp_path)
                final_result["diagnosis"]    = prediction.get("predicted_class", "UNKNOWN")
                final_result["confidence"]   = prediction.get("confidence", 0.0)
                final_result["probabilities"] = prediction.get("probabilities", final_result["probabilities"])
                final_result["risk"] = "LOW" if final_result["diagnosis"] == "NORMAL" else "HIGH"
                final_result["recommendation"] = (
                    "Patient appears healthy. Routine follow-up." if final_result["diagnosis"] == "NORMAL"
                    else f"Findings consistent with {final_result['diagnosis']}. Clinical correlation required."
                )
            except Exception as e:
                print(f"Prediction error: {e}")

            # ---- 3. Uncertainty (MC Dropout) ----
            try:
                uncertainty = estimate_uncertainty(temp_path, num_passes=10)
                final_result["uncertainty"] = uncertainty.get("uncertainty_score", final_result["uncertainty"])
            except Exception as e:
                print(f"Uncertainty error: {e}")

            final_result["needs_review"] = float(final_result["uncertainty"]) > UNCERTAINTY_THRESHOLD

            # ---- 4. Grad-CAM ----
            try:
                gradcam_filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
                gradcam_path = RESULTS_DIR / gradcam_filename
                generate_gradcam(temp_path, output_path=str(gradcam_path))
                # Relative path — served by the /results mount below
                final_result["gradcam_url"] = f"/results/{gradcam_filename}"
            except Exception as e:
                print(f"Grad-CAM error: {e}")

            # ---- 5. Similar images ----
            try:
                similar = find_similar_images(temp_path, top_k=5)
                similar_urls = []
                for item in (similar or []):
                    if isinstance(item, dict):
                        src_path = item.get("image_path") or item.get("path") or item.get("url")
                    else:
                        src_path = str(item)
                    if not src_path or not os.path.exists(src_path):
                        continue
                    ext = Path(src_path).suffix or ".jpg"
                    dest_name = f"sim_{uuid.uuid4().hex[:8]}{ext}"
                    dest = RESULTS_DIR / "similar" / dest_name
                    dest.parent.mkdir(exist_ok=True)
                    try:
                        shutil.copy(src_path, dest)
                        similar_urls.append(f"/results/similar/{dest_name}")
                    except Exception as ce:
                        print(f"Similar copy error: {ce}")
                final_result["similar"] = similar_urls
                print(f"Similar images copied: {len(similar_urls)}")
            except Exception as e:
                print(f"Similarity error: {e}")

            print(f"✅ Prediction: {final_result['diagnosis']} "
                  f"({final_result['confidence']}%) "
                  f"unc={final_result['uncertainty']}%")

        # ---- Save to DB ----
        db = SessionLocal()
        try:
            db.add(PatientRecord(
                name=patient_name,
                age=age,
                symptoms=symptoms,
                diagnosis=final_result["diagnosis"],
                confidence=float(final_result["confidence"]),
            ))
            db.commit()
        except Exception as e:
            print(f"DB error: {e}")
            db.rollback()
        finally:
            db.close()

        return final_result

    except Exception as e:
        return {"error": str(e)}
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except OSError: pass


@app.get("/patients/history")
def get_history():
    db = SessionLocal()
    try:
        records = db.query(PatientRecord).order_by(PatientRecord.id.desc()).all()
        return [
            {
                "id": r.id, "name": r.name, "age": r.age,
                "diagnosis": r.diagnosis, "confidence": r.confidence,
                "date": r.date.strftime("%Y-%m-%d %H:%M")
            }
            for r in records
        ]
    finally:
        db.close()


@app.get("/analytics")
def get_analytics():
    db = SessionLocal()
    try:
        records = db.query(PatientRecord).all()
    finally:
        db.close()

    disease_counts = {"COVID19": 0, "NORMAL": 0, "PNEUMONIA": 0, "TUBERCULOSIS": 0}
    age_groups = {"0-18": 0, "19-35": 0, "36-50": 0, "50+": 0}
    high_risk = 0
    recent = []

    for r in records:
        if r.diagnosis in disease_counts:
            disease_counts[r.diagnosis] += 1
        age = r.age or 0
        if age <= 18: age_groups["0-18"] += 1
        elif age <= 35: age_groups["19-35"] += 1
        elif age <= 50: age_groups["36-50"] += 1
        else: age_groups["50+"] += 1
        if (r.confidence or 0) > 80 and r.diagnosis != "NORMAL":
            high_risk += 1
        recent.append({
            "date": r.date.strftime("%Y-%m-%d %H:%M"),
            "name": r.name, "diagnosis": r.diagnosis, "confidence": r.confidence
        })

    recent.sort(key=lambda x: x["date"], reverse=True)
    return {
        "total_patients": len(records),
        "high_risk": high_risk,
        "disease_counts": disease_counts,
        "age_groups": age_groups,
        "recent_activity": recent[:5]
    }


# Serve Grad-CAM images
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

@app.get("/model/comparison")
def model_comparison():
    """
    Model comparison metrics.
    Try to read results/comparison.json (generated by comparison.py).
    Fall back to training-time evaluation values.
    """
    import json
    path = Path("results/comparison.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)

    # Fallback: from your training logs
    return {
        "models": [
            {
                "name": "EfficientNet-B0",
                "accuracy": 89.36, "precision": 88.7, "recall": 87.9, "f1_score": 88.3,
                "parameters": "5.3M", "inference_time_ms": 42,
                "is_production": True,
                "per_class": {
                    "COVID19": {"precision": 91.2, "recall": 89.5, "f1": 90.3},
                    "NORMAL": {"precision": 94.8, "recall": 96.1, "f1": 95.4},
                    "PNEUMONIA": {"precision": 87.3, "recall": 85.8, "f1": 86.5},
                    "TUBERCULOSIS": {"precision": 81.4, "recall": 80.2, "f1": 80.8}
                }
            },
            {
                "name": "ResNet18",
                "accuracy": 84.12, "precision": 83.5, "recall": 82.1, "f1_score": 82.8,
                "parameters": "11.7M", "inference_time_ms": 38,
                "is_production": False,
                "per_class": {
                    "COVID19": {"precision": 85.7, "recall": 83.2, "f1": 84.4},
                    "NORMAL": {"precision": 91.2, "recall": 92.8, "f1": 92.0},
                    "PNEUMONIA": {"precision": 82.1, "recall": 79.5, "f1": 80.8},
                    "TUBERCULOSIS": {"precision": 75.3, "recall": 72.8, "f1": 74.0}
                }
            }
        ],
        "winner": "EfficientNet-B0",
        "summary": "EfficientNet-B0 outperforms ResNet18 by 5.24% accuracy with 55% fewer parameters."
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
