
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pathlib import Path
# import tempfile
# import os

# from src.predict import predict_image

# app = FastAPI(
#     title="MedVision AI API",
#     description="Chest X-ray classification API",
#     version="1.0"
# )


# @app.get("/")
# def home():
#     return {
#         "message": "Welcome to MedVision AI API",
#         "status": "Running"
#     }


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):

#     allowed_types = [
#         "image/jpeg",
#         "image/png"
#     ]

#     if file.content_type not in allowed_types:
#         raise HTTPException(
#             status_code=400,
#             detail="Please upload a JPG or PNG image."
#         )

#     content = await file.read()

#     if not content:
#         raise HTTPException(
#             status_code=400,
#             detail="Uploaded image is empty."
#         )

#     if len(content) > 10 * 1024 * 1024:
#         raise HTTPException(
#             status_code=413,
#             detail="Image must be smaller than 10 MB."
#         )

#     temp_path = None

#     try:
#         with tempfile.NamedTemporaryFile(
#             suffix=Path(file.filename or "image.jpg").suffix,
#             delete=False
#         ) as temp:
#             temp.write(content)
#             temp_path = temp.name

#         result = predict_image(temp_path)

#         return {
#             "filename": file.filename,
#             "prediction": result
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=400,
#             detail=str(e)
#         )

#     finally:
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)



from src.similarity import find_similar_images
from src.uncertainty import estimate_uncertainty

from src.image_quality import check_image_quality
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import os

from src.predict import predict_image
from src.gradcam import generate_gradcam

app = FastAPI(
    title="MedVision AI API",
    description="Explainable Chest X-ray Classification API",
    version="1.1"
)

RESULT_PATH = Path("results/gradcam_latest.png")


@app.get("/")
def home():
    return {
        "message": "Welcome to MedVision AI API",
        "status": "Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_types = ["image/jpeg", "image/png"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty image.")

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large.")

    temp_path = None

    try:
        suffix = Path(file.filename or "image.jpg").suffix.lower()

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp:
            temp.write(content)
            temp_path = temp.name

        result = predict_image(temp_path)

        return {
            "filename": file.filename,
            "prediction": result
        }

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to process the uploaded image."
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/explain")
async def explain(file: UploadFile = File(...)):

    allowed_types = ["image/jpeg", "image/png"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty image.")

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large.")

    temp_path = None

    try:
        suffix = Path(file.filename or "image.jpg").suffix.lower()

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp:
            temp.write(content)
            temp_path = temp.name

        # Generate Grad-CAM heatmap
        result = generate_gradcam(
            temp_path,
            output_path=RESULT_PATH
        )

        return {
            "filename": file.filename,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "gradcam_image_url": "/gradcam-image"
        }

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to generate Grad-CAM explanation."
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/gradcam-image")
def get_gradcam_image():

    if not RESULT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Generate a Grad-CAM image first."
        )

    return FileResponse(
        RESULT_PATH,
        media_type="image/png",
        filename="gradcam.png"
    )

@app.post("/check-quality")
async def check_quality(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB."
        )

    temp_path = None

    try:
        suffix = Path(file.filename or "image.jpg").suffix.lower()

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp:
            temp.write(content)
            temp_path = temp.name

        result = check_image_quality(temp_path)

        return {
            "filename": file.filename,
            "quality_assessment": result
        }

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image."
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# Combined Image Quality and Classification Endpoint

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):

    # Check uploaded file type
    allowed_types = ["image/jpeg", "image/png"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    # Read uploaded image
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    # Maximum file size: 10 MB
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB."
        )

    temp_path = None

    try:
        # Save uploaded image temporarily
        suffix = Path(
            file.filename or "image.jpg"
        ).suffix.lower()

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp:
            temp.write(content)
            temp_path = temp.name

        # Step 1: Check image quality
        quality_result = check_image_quality(temp_path)

        # Step 2: Predict image class
        prediction_result = predict_image(temp_path)
        uncertainty_result = estimate_uncertainty(temp_path, num_passes=20)
        similar_images = find_similar_images(temp_path, top_k=5)

        #Step 3: Return combined results
        return {
            "filename": file.filename,
            "quality_assessment": quality_result,
            "prediction": prediction_result,
            "uncertainty_assessment": uncertainty_result,
            "similar_images": similar_images,
            "review_required": (
                quality_result["quality_status"]
                == "Needs Review"
            ),
            "message": (
                "Analysis completed. Review image quality warnings."
                if quality_result["quality_status"] == "Needs Review"
                else "Analysis completed."
            )
        }

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image."
        )

    finally:
        # Delete temporary image
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/uncertainty")
async def uncertainty_endpoint(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB."
        )

    temp_path = None

    try:
        suffix = Path(file.filename or "image.jpg").suffix.lower()

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp:
            temp.write(content)
            temp_path = temp.name

        result = estimate_uncertainty(
            temp_path,
            num_passes=20
        )

        return {
            "filename": file.filename,
            "uncertainty_assessment": result
        }

    except (ValueError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image."
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)





