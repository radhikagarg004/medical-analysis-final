from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import uuid
from pathlib import Path
from src.api.database import SessionLocal, PatientRecord

# 🔥 Radhika ka real AI import karo
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

app = FastAPI(title="MediScan AI API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Grad-CAM images save karne ke liye folder
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

@app.get("/health")
def health_check():
    return {"status": "ok", "ai_mode": "REAL" if AI_READY else "MOCK"}

@app.post("/predict/diagnosis")
async def predict_diagnosis(
    file: UploadFile = File(...),
    patient_name: str = Form(...),
    age: int = Form(...),
    symptoms: str = Form("")
):
    temp_path = None
    try:
        suffix = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(await file.read())
            temp_path = temp.name

        final_result = {
            "diagnosis": "PNEUMONIA (Mock)",
            "confidence": 93.4,
            "risk": "HIGH",
            "opacity": 18.5,
            "recommendation": "Please consult a doctor.",
            "probabilities": {"COVID19": 5, "NORMAL": 10, "PNEUMONIA": 80, "TUBERCULOSIS": 5},
            "uncertainty": 2.5,
            "quality": "Acceptable",
            "gradcam_url": None,
            "similar_images": []
        }

        if AI_READY:
            try:
                print("🧠 Running Real AI...")
                
                # 1. Quality Check
                quality = check_image_quality(temp_path)
                final_result["quality"] = quality.get("quality_status", "Unknown")
                
                # 2. Prediction
                prediction = predict_image(temp_path)
                final_result["diagnosis"] = prediction["predicted_class"]
                final_result["confidence"] = prediction["confidence"]
                final_result["probabilities"] = prediction["probabilities"]
                final_result["risk"] = "HIGH" if prediction["predicted_class"] != "NORMAL" else "LOW"
                final_result["recommendation"] = "Consult a doctor immediately." if prediction["predicted_class"] != "NORMAL" else "Patient is healthy."
                
                # 3. Uncertainty
                uncertainty = estimate_uncertainty(temp_path, num_passes=5)
                final_result["uncertainty"] = uncertainty.get("uncertainty_score", 2.5)
                
                # 4. Grad-CAM (Real heatmap)
                try:
                    gradcam_filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
                    gradcam_path = RESULTS_DIR / gradcam_filename
                    generate_gradcam(temp_path, output_path=str(gradcam_path))
                    final_result["gradcam_url"] = f"http://localhost:8001/results/{gradcam_filename}"
                except Exception as ge:
                    print(f"Grad-CAM error: {ge}")
                
                # 5. Similar Images
                try:
                    similar = find_similar_images(temp_path, top_k=3)
                    final_result["similar_images"] = similar
                except Exception as se:
                    print(f"Similarity error: {se}")
                
                print(f"✅ Real Prediction: {prediction['predicted_class']} ({prediction['confidence']}%)")
            except Exception as e:
                print(f"⚠️ AI error: {e}. Using mock.")

        # Database save
        db = SessionLocal()
        try:
            new_record = PatientRecord(
                name=patient_name, age=age, symptoms=symptoms,
                diagnosis=final_result["diagnosis"], confidence=final_result["confidence"]
            )
            db.add(new_record)
            db.commit()
        finally:
            db.close()

        return final_result

    except Exception as e:
        return {"error": str(e)}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/patients/history")
def get_history():
    db = SessionLocal()
    records = db.query(PatientRecord).order_by(PatientRecord.id.desc()).all()
    db.close()
    return [
        {
            "id": r.id, "name": r.name, "age": r.age,
            "diagnosis": r.diagnosis, "confidence": r.confidence,
            "date": r.date.strftime("%Y-%m-%d %H:%M")
        }
        for r in records
    ]

# 🔥 NAYA: Analytics Endpoint
@app.get("/analytics")
def get_analytics():
    db = SessionLocal()
    records = db.query(PatientRecord).all()
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
        
        if r.confidence > 80 and r.diagnosis != "NORMAL":
            high_risk += 1
        
        recent.append({
            "date": r.date.strftime("%Y-%m-%d %H:%M"),
            "name": r.name,
            "diagnosis": r.diagnosis,
            "confidence": r.confidence
        })
    
    recent.sort(key=lambda x: x["date"], reverse=True)
    
    return {
        "total_patients": len(records),
        "high_risk": high_risk,
        "disease_counts": disease_counts,
        "age_groups": age_groups,
        "recent_activity": recent[:5]
    }

# 🔥 NAYA: Grad-CAM images serve karne ke liye
from fastapi.staticfiles import StaticFiles
app.mount("/results", StaticFiles(directory="results"), name="results")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)