from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import onnxruntime as ort
from onnxruntime import SessionOptions, GraphOptimizationLevel
import multiprocessing
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os
from typing import Optional
from contextlib import asynccontextmanager
import static.anodet as anodet


# Global variables
sess = None
ANOMALY_THRESHOLD = 13.0
RESIZE_SIZE = (224, 224)

async def load_model():
    global sess, padim_model
    try:
        # Get the list of available execution providers (e.g., CPUExecutionProvider, CUDAExecutionProvider)
        available_providers = ort.get_available_providers()
        # Check if CUDA (GPU) is available by looking for the CUDAExecutionProvider
        use_gpu = "CUDAExecutionProvider" in available_providers
        
        # Create session options for ONNX Runtime
        sess_options = SessionOptions()

        # Enable memory pattern optimization to improve performance on repeated inference calls
        sess_options.enable_mem_pattern = True

        # Enable graph optimization to apply advanced model graph transformations
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # These two options are useful only when using CPU
        if not use_gpu:
            # Enable memory arena for CPU allocator to optimize memory usage
            sess_options.enable_cpu_mem_arena = True

            # Set the number of threads ONNX Runtime can use for CPU operations
            sess_options.intra_op_num_threads = multiprocessing.cpu_count()

        # Set the preferred execution provider based on hardware availability
        providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]        
        
        # Try to load ONNX model first
        if os.path.exists("padim_model.onnx"):
            sess = ort.InferenceSession("padim_model.onnx", providers=providers, sess_options=sess_options)                                
            print("ONNX model loaded successfully.")
        else:
            raise FileNotFoundError("No model found. Please ensure 'padim_model.onnx' ")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

async def cleanup():
    global sess
    sess = None
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
    anomaly_map_base64: Optional[str] = ""
    boundary_image_base64: Optional[str] = ""
    heatmap_image_base64: Optional[str] = ""
    highlighted_image_base64: Optional[str] = ""

class ConfigModel(BaseModel):
    threshold: float = ANOMALY_THRESHOLD
    resize_width: int = 224
    resize_height: int = 224

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
    model_loaded = sess is not None 
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_type": "onnx" if sess else "pytorch" if padim_model else "none",
        "threshold": ANOMALY_THRESHOLD,
        "resize_size": RESIZE_SIZE
    }

@app.post("/config")
async def update_config(config: ConfigModel):
    global ANOMALY_THRESHOLD, RESIZE_SIZE
    ANOMALY_THRESHOLD = config.threshold
    RESIZE_SIZE = (config.resize_width, config.resize_height)
    return {
        "message": f"Threshold updated to {ANOMALY_THRESHOLD}, Resize size set to {RESIZE_SIZE}"
    }

def preprocess_image_from_upload(file_contents: bytes) -> np.ndarray:
    """Convert uploaded file to numpy array matching detect.py preprocessing"""
    # Read image using PIL and convert to RGB
    image_pil = Image.open(io.BytesIO(file_contents)).convert("RGB")
    
    # Convert PIL to numpy array (this matches how detect.py reads with cv2 then converts to RGB)
    image_np = np.array(image_pil)
    
    return image_np

def create_visualizations(image_np: np.ndarray, score_maps: np.ndarray, image_scores: np.ndarray) -> tuple:
    """Create visualization images using NumPy arrays (torch-free)."""
    try:
        print("Creating visualizations...")

        # Apply classification using NumPy
        score_map_classifications = anodet.classification(score_maps, ANOMALY_THRESHOLD)
        image_classifications = anodet.classification(image_scores, ANOMALY_THRESHOLD)

        # Prepare image array
        test_images = np.array([image_np])

        # Create visualizations
        boundary_images = anodet.visualization.framed_boundary_images(
            test_images,
            score_map_classifications,
            image_classifications,
            padding=40
        )

        heatmap_images = anodet.visualization.heatmap_images(
            test_images,
            score_maps,
            alpha=0.5
        )

        highlighted_images = anodet.visualization.highlighted_images(
            [image_np],
            score_map_classifications,
            color=(128, 0, 128)
        )

        print("All visualizations created successfully")

        return boundary_images[0], heatmap_images[0], highlighted_images[0]

    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def numpy_to_base64(image_array: np.ndarray, resize_to: tuple[int, int] = None) -> str:
    """Convert numpy array to base64-encoded PNG, optionally resized"""
    if image_array is None:
        return ""
    
    try:
        # Ensure the array is in the right format (0-255 uint8)
        if image_array.dtype != np.uint8:
            # Handle different data ranges
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(image_array)
        if resize_to is not None:
            image = image.resize(resize_to, Image.BILINEAR)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error converting to base64: {e}")
        return ""

@app.post("/predict", response_model=PredictionResult)
async def predict_anomaly(
    file: UploadFile = File(...),
    include_visualizations: bool = True
):
    if sess is None:
        raise HTTPException(status_code=500, detail="No model loaded.")

    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and preprocess image exactly like detect.py
        contents = await file.read()
        image_np = preprocess_image_from_upload(contents)
        
        print(f"Image shape: {image_np.shape}")
        print(f"Image dtype: {image_np.dtype}")
        
        if sess is not None:
            # ONNX inference - convert single image to batch
            batch = anodet.to_batch([image_np])
            input_numpy = batch
            
            input_name = sess.get_inputs()[0].name
            output_names = [output.name for output in sess.get_outputs()]
            
            if len(output_names) < 2:
                raise HTTPException(status_code=500, detail="Model must have at least 2 outputs")
            
            outputs = sess.run(output_names, {input_name: input_numpy})            
            
            image_scores = np.array([outputs[0]])
            score_maps = np.array(outputs[1])
            

        print(f"Image scores: {image_scores}")
        print(f"Score maps shape: {score_maps.shape}")
        
        # Get single values
        anomaly_score = float(image_scores[0])
        is_anomaly = anomaly_score >= ANOMALY_THRESHOLD
        
        # Create anomaly map (normalized score map)
        # score_map_np = score_maps[0]
        # score_map_normalized = score_map_np.copy()
        
        # if score_map_np.max() - score_map_np.min() > 0:
        #     score_map_normalized = (score_map_np - score_map_np.min()) / (score_map_np.max() - score_map_np.min())
        # else:
        #     score_map_normalized = np.zeros_like(score_map_np)

        # anomaly_map_base64 = numpy_to_base64(score_map_normalized,RESIZE_SIZE)

        # Initialize visualization base64 strings
        boundary_image_base64 = ""
        heatmap_image_base64 = ""
        highlighted_image_base64 = ""

        if include_visualizations:
            print("Creating visualizations...")
            boundary_image, heatmap_image, highlighted_image = create_visualizations(
                image_np, score_maps, image_scores
            )

            if boundary_image is not None:
                boundary_image_base64 = numpy_to_base64(boundary_image,RESIZE_SIZE)
                print("✓ Boundary image created")
            else:
                print("❗ Boundary image is None")

            if heatmap_image is not None:
                heatmap_image_base64 = numpy_to_base64(heatmap_image,RESIZE_SIZE)
                print("✓ Heatmap image created")
            else:
                print("❗ Heatmap image is None")
                
            # if highlighted_image is not None:
            #     highlighted_image_base64 = numpy_to_base64(highlighted_image,RESIZE_SIZE)
            #     print("✓ Highlighted image created")
            # else:
            #     print("❗ Highlighted image is None")

        return PredictionResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            # anomaly_map_base64=anomaly_map_base64,
            boundary_image_base64=boundary_image_base64,
            heatmap_image_base64=heatmap_image_base64,
            # highlighted_image_base64=highlighted_image_base64
        )

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    for i, file in enumerate(files):
        try:
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
        
    else:
        raise HTTPException(status_code=500, detail="No model loaded")

if __name__ == "__main__":
    # Use string import for proper reload functionality
    uvicorn.run("fastapi_app_np:app", host="0.0.0.0", port=8080, reload=False)
# run that from the Docker image,  see docker\Dockerfile.np   


