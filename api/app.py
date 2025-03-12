# app.py
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from PIL import Image
import uvicorn
import logging
from pydantic import BaseModel
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("image_classifier_api")

# Initialize FastAPI app
app = FastAPI(title="Image Classification API", 
              description="API for classifying images using a pre-trained model",
              version="1.0.0")

# Security
security = HTTPBasic()

# Define valid credentials (in production, use a more secure approach)
USERNAME = "admin"
PASSWORD = "password123"

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify the user's credentials."""
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    
    if not (correct_username and correct_password):
        logger.warning(f"Failed authentication attempt: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username

# Load the trained model
try:
    logger.info("Loading model...")
    MODEL_PATH = "models/intel_classifier_deployment"
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(f"{MODEL_PATH}/class_names.npy", allow_pickle=True)
    logger.info(f"Model loaded successfully with classes: {class_names}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Define response models
class Prediction(BaseModel):
    class_name: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]
    top_prediction: str

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Image Classification API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    """
    Predict the class of an uploaded image.
    
    - **file**: The image file to classify
    """
    logger.info(f"Prediction request received from user: {username}")
    
    # Validate file
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess the image
        image = image.convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = model.predict(image_array)[0]
        
        # Get top 3 predictions
        top_indices = predictions.argsort()[-3:][::-1]
        top_predictions = [
            Prediction(
                class_name=class_names[idx],
                confidence=float(predictions[idx])
            )
            for idx in top_indices
        ]
        
        # Log the prediction
        logger.info(f"Predicted class: {class_names[top_indices[0]]} with confidence: {predictions[top_indices[0]]:.4f}")
        
        return PredictionResponse(
            predictions=top_predictions,
            top_prediction=class_names[top_indices[0]]
        )
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)