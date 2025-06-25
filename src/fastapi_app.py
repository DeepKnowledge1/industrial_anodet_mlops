from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import onnxruntime
import numpy as np
from PIL import Image
import io
import base64
import torch
import cv2
import os
from typing import Optional
from contextlib import asynccontextmanager
import AnomaVision.anodet as anodet


# Global variables
sess = None
padim_model = None
ANOMALY_THRESHOLD = 13.0  # Based on your detect.py

async def load_model():
    global sess, padim_model
    try:
        # Try to load ONNX model first
        if os.path.exists("padim_model.onnx"):
            sess = onnxruntime.InferenceSession("padim_model.onnx")
            print("ONNX model loaded successfully.")
        # Fallback to PyTorch model
        elif os.path.exists("distributions/padim_model.pt"):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            padim_model = torch.load("distributions/padim_model.pt", map_location=device)
            padim_model.eval()
            print("PyTorch model loaded successfully.")
        else:
            raise FileNotFoundError("No model found. Please ensure 'padim_model.onnx' or 'distributions/padim_model.pt' exists.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

async def cleanup():
    global sess, padim_model
    sess = None
    padim_model = None
    print("Model cleanup completed.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model()
    yield
    # Shutdown
    await cleanup()

app = FastAPI(title="Anomaly Detection API", version="1.0.0", lifespan=lifespan)

class PredictionResult(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    anomaly_map_base64: str
    boundary_image_base64: Optional[str] = None
    heatmap_image_base64: Optional[str] = None

class ConfigModel(BaseModel):
    threshold: float = ANOMALY_THRESHOLD



@app.get("/")
async def root():
    return {
        "message": "Anomaly Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict-batch",
            "model_info": "/model-info",
            "config": "/config",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health_check():
    model_loaded = sess is not None or padim_model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_type": "onnx" if sess else "pytorch" if padim_model else "none",
        "threshold": ANOMALY_THRESHOLD
    }

@app.post("/config")
async def update_config(config: ConfigModel):
    global ANOMALY_THRESHOLD
    ANOMALY_THRESHOLD = config.threshold
    return {"message": f"Threshold updated to {ANOMALY_THRESHOLD}"}

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert PIL image to numpy array in the format expected by anodet"""
    # Convert PIL to numpy array (RGB)
    image_np = np.array(image)
    return image_np

def create_visualizations(image_np: np.ndarray, score_map: np.ndarray, 
                         image_score: float) -> tuple:
    """Create visualization images similar to detect.py"""
    try:
        # Convert score map to classification
        score_map_classification = anodet.classification(score_map, ANOMALY_THRESHOLD)
        image_classification = anodet.classification(image_score, ANOMALY_THRESHOLD)
        
        # Create boundary image
        boundary_image = anodet.visualization.framed_boundary_images(
            np.array([image_np]), 
            [score_map_classification], 
            [image_classification], 
            padding=40
        )[0]
        
        # Create heatmap image
        heatmap_image = anodet.visualization.heatmap_images(
            np.array([image_np]), 
            [score_map], 
            alpha=0.5
        )[0]
        
        return boundary_image, heatmap_image
    except Exception as e:
        print(f"Visualization error: {e}")
        return None, None

def numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG"""
    if image_array is None:
        return ""
    
    # Ensure the array is in the right format
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Convert to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/predict", response_model=PredictionResult)
async def predict_anomaly(
    file: UploadFile = File(...),
    include_visualizations: bool = True
):
    if sess is None and padim_model is None:
        raise HTTPException(status_code=500, detail="No model loaded.")

    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = preprocess_image(image)
        
        if sess is not None:
            # ONNX inference
            input_tensor = anodet.standard_image_transform(image).unsqueeze(0)
            input_numpy = input_tensor.numpy()
            
            input_name = sess.get_inputs()[0].name
            output_names = [output.name for output in sess.get_outputs()]
            
            if len(output_names) < 2:
                raise HTTPException(status_code=500, detail="Model must have at least 2 outputs")
            
            outputs = sess.run(output_names, {input_name: input_numpy})
            anomaly_score = float(outputs[0].item())
            score_map = outputs[1].squeeze()
            
        else:
            # PyTorch inference
            batch = anodet.to_batch([image_np], anodet.standard_image_transform, padim_model.device)
            with torch.no_grad():
                image_scores, score_maps = padim_model.predict(batch)
            
            anomaly_score = float(image_scores[0])
            score_map = score_maps[0].cpu().numpy()

        # Determine if it's an anomaly
        is_anomaly = anomaly_score > ANOMALY_THRESHOLD
        
        # Create anomaly map visualization
        score_map_normalized = score_map.copy()
        if score_map.max() - score_map.min() > 0:
            score_map_normalized = (score_map - score_map.min()) / (score_map.max() - score_map.min())
        else:
            score_map_normalized = np.zeros_like(score_map)
        
        anomaly_map_base64 = numpy_to_base64(score_map_normalized)
        
        # Create additional visualizations if requested
        boundary_image_base64 = None
        heatmap_image_base64 = None
        
        if include_visualizations:
            try:
                boundary_image, heatmap_image = create_visualizations(
                    image_np, score_map, anomaly_score
                )
                if boundary_image is not None:
                    boundary_image_base64 = numpy_to_base64(boundary_image)
                if heatmap_image is not None:
                    heatmap_image_base64 = numpy_to_base64(heatmap_image)
            except Exception as viz_error:
                print(f"Visualization creation failed: {viz_error}")
                # Continue without visualizations

        return PredictionResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            anomaly_map_base64=anomaly_map_base64,
            boundary_image_base64=boundary_image_base64,
            heatmap_image_base64=heatmap_image_base64
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="Model file not found")
    except onnxruntime.OrtException as e:
        raise HTTPException(status_code=500, detail=f"ONNX Runtime error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch prediction endpoint"""
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    for i, file in enumerate(files):
        try:
            # Process each file
            result = await predict_anomaly(file, include_visualizations=False)
            results.append({
                "file_index": i,
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "file_index": i,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"batch_results": results}

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if sess is not None:
        inputs = [(inp.name, inp.shape, inp.type) for inp in sess.get_inputs()]
        outputs = [(out.name, out.shape, out.type) for out in sess.get_outputs()]
        return {
            "model_type": "onnx",
            "inputs": inputs,
            "outputs": outputs,
            "threshold": ANOMALY_THRESHOLD
        }
    elif padim_model is not None:
        return {
            "model_type": "pytorch",
            "device": str(padim_model.device),
            "threshold": ANOMALY_THRESHOLD
        }
    else:
        raise HTTPException(status_code=500, detail="No model loaded")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
        
    # To run use :  uvicorn src/fastapi_app:app --host 0.0.0.0 --port 8000
