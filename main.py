from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import json
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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code: load existing images
    print("Loading existing images from disk and running model inference...")

    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)

        if not os.path.isfile(file_path) or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # check if already in db
        if any(img["filename"] == filename for img in image_store):
            continue

        try:
            img = image.load_img(file_path, target_size=(299, 299))
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)  # Use InceptionResNetV2 preprocessing
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            prediction = model.predict(img_array)
            predicted_class = "OK" if prediction[0][0] > 0.5 else "NOK"

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

CONFIG_PATH = "configurations/config.json"  # Path to custom config file
DEFAULT_CONFIG_PATH = "configurations/default.json"  # Path to default config file

# Load configuration from file
def load_config():
    if not os.path.exists(CONFIG_PATH):
        # Default configuration structure
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            default_config = json.load(f)
            with open(CONFIG_PATH, "w") as f2:
                json.dump(default_config, f2, indent=2)
            return default_config
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

# Save configuration to file
def save_config(config_data: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_data, f, indent=2)

@app.get("/config")
async def get_config():
    try:
        config = load_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/config")
async def update_config(new_config: dict):
    try:
        # Optionally: validate fields here
        save_config(new_config)
        return JSONResponse(content={"message": "Configuration updated."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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
