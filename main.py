from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List
import shutil
import os
import uuid
import tensorflow as tf
import numpy as np
from PIL import Image

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code: load existing images
    print("Loading existing images from disk and running model inference...")

    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)

        if not os.path.isfile(file_path) or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        if any(img["filename"] == filename for img in image_store):
            continue

        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = "NOK" if prediction[0][0] > 0.5 else "OK"

            image_store.append({"filename": filename, "result": predicted_class})
            print(f"Loaded {filename} - {predicted_class}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    yield

app = FastAPI(lifespan=lifespan)

# Allow CORS for development (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded images
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load Keras model (assumes model is in the same directory)
model = tf.keras.models.load_model("models/classifier.keras")

# Store results in memory (replace with DB for production)
image_store = []

class ImageResult(BaseModel):
    url: str



app.mount("/uploads", StaticFiles(directory="uploaded_images"), name="uploads")
@app.post("/upload", response_model=ImageResult)
async def upload_image(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and preprocess image for model
        img = Image.open(file_path).convert("RGB")
        img = img.resize((224, 224))  # adjust to your model's input size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        prediction = model.predict(img_array)
        predicted_class = "NOK" if prediction[0][0] > 0.5 else "OK"  # adjust logic as per your model

        # Store result
        image_store.append({"filename": unique_filename, "result": predicted_class})

        return {"filename": unique_filename, "result": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/nok", response_model=List[ImageResult])
async def get_nok_images(request: Request):
    nok_images = [
        {
            "url": str(request.base_url) + f"uploads/{img['filename']}"
        }
        for img in image_store if img["result"] == "NOK"
    ]
    return nok_images

@app.get("/images/all", response_model=List[ImageResult])
async def get_nok_images(request: Request):
    nok_images = [
        {
            "url": str(request.base_url) + f"uploads/{img['filename']}"
        }
        for img in image_store
    ]
    return nok_images
