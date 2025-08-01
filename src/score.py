import json
import numpy as np
import os
import onnxruntime as ort
from onnxruntime import SessionOptions, GraphOptimizationLevel
import multiprocessing
from PIL import Image
import io
import base64
import cv2
import logging
import time

import static.anodet as anodet

# Configure logging

from logger import setup_logging

setup_logging()
# Now use the logger
logger = logging.getLogger("industrial-mlops")




# Global variables for model and configuration
sess = None
ANOMALY_THRESHOLD = 13.0
RESIZE_SIZE = (224, 224)

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can set global variables here that are needed by your run() function.
    """
    global sess, ANOMALY_THRESHOLD, RESIZE_SIZE
    
    # Load configuration from config.json if it exists
    config_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            ANOMALY_THRESHOLD = config.get("threshold", ANOMALY_THRESHOLD)
            resize_width = config.get("resize_width", RESIZE_SIZE[0])
            resize_height = config.get("resize_height", RESIZE_SIZE[1])
            RESIZE_SIZE = (resize_width, resize_height)
            logger.info(f"Loaded config: threshold={ANOMALY_THRESHOLD}, resize_size={RESIZE_SIZE}")

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
        
        # Model path for Azure ML deployment
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "padim_model.onnx")
        model_path = "./model_output/padim_model.onnx" ## YOU HAVE TO DELETE IT
        
        # Try to load ONNX model
        if os.path.exists(model_path):
            sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)                                
            logger.info("ONNX model loaded successfully.")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}. Please ensure \'padim_model.onnx\' is in the model directory.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def preprocess_image_from_bytes_PIL(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to numpy array matching detect.py preprocessing"""
    # Read image using PIL and convert to RGB
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Convert PIL to numpy array
    image_np = np.array(image_pil)
    
    return image_np

def preprocess_image_from_bytes_CV(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to numpy array using OpenCV"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode image
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # IMREAD_COLOR for BGR
    if image_np is None:
        raise ValueError("Could not decode image.")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    return image_np

def create_visualizations(image_np: np.ndarray, score_maps: np.ndarray, image_scores: np.ndarray) -> tuple:
    """Create visualization images using NumPy arrays (torch-free)."""
    try:
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

        # highlighted_images = anodet.visualization.highlighted_images(
        #     [image_np],
        #     score_map_classifications,
        #     color=(128, 0, 128)
        # )

        return boundary_images[0], heatmap_images[0], None # highlighted_images[0]

    except Exception as e:
        logger.error(f"Visualization error: {e}")
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
        logger.error(f"Error converting to base64: {e}")
        return ""

def run(raw_data):
    """
    This function is called for every incoming request.
    """
    total_start_time = time.time() # Start total timing
    try:
        data = json.loads(raw_data)
        image_base64 = data.get("image_base64")
        include_visualizations = data.get("include_visualizations", True)

        if not image_base64:
            logger.warning("No image_base64 provided in the request.")
            return json.dumps({"error": "No image_base64 provided"})

        # Preprocessing time
        preprocess_start_time = time.time()
        image_bytes = base64.b64decode(image_base64)
        image_np = preprocess_image_from_bytes_CV(image_bytes)
        preprocess_end_time = time.time()
        preprocess_time = (preprocess_end_time - preprocess_start_time) * 1000

        # Image Analysis Metrics
        img_height, img_width, img_channels = image_np.shape
        img_pixel_mean = np.mean(image_np)
        img_pixel_std = np.std(image_np)
        img_pixel_min = np.min(image_np)
        img_pixel_max = np.max(image_np)
        img_total_pixels = image_np.size
        logger.info(f"Image Analysis Metrics: Dimensions: ({img_height}, {img_width}, {img_channels}), Pixel Stats (Mean/Std/Min/Max): {img_pixel_mean:.2f}/{img_pixel_std:.2f}/{img_pixel_min:.2f}/{img_pixel_max:.2f}, Total Pixels: {img_total_pixels}")

        if sess is None:
            logger.error("Model not loaded. Cannot perform inference.")
            return json.dumps({"error": "Model not loaded."})

        # Inference time
        inference_start_time = time.time()
        batch = anodet.to_batch([image_np])
        input_numpy = batch
        
        input_name = sess.get_inputs()[0].name
        output_names = [output.name for output in sess.get_outputs()]
        
        if len(output_names) < 2:
            logger.error("Model must have at least 2 outputs for anomaly detection.")
            return json.dumps({"error": "Model must have at least 2 outputs"})
        
        outputs = sess.run(output_names, {input_name: input_numpy})            
        
        image_scores = np.array([outputs[0]])
        score_maps = np.array(outputs[1])
        inference_end_time = time.time()
        inference_time = (inference_end_time - inference_start_time) * 1000
        
        anomaly_score = float(image_scores[0])
        is_anomaly = anomaly_score >= ANOMALY_THRESHOLD
        
        boundary_image_base64 = ""
        heatmap_image_base64 = ""
        highlighted_image_base64 = ""

        visualization_time = 0
        if include_visualizations:
            visualization_start_time = time.time()
            boundary_image, heatmap_image, highlighted_image = create_visualizations(
                image_np, score_maps, image_scores
            )

            if boundary_image is not None:
                boundary_image_base64 = numpy_to_base64(boundary_image, RESIZE_SIZE)
            if heatmap_image is not None:
                heatmap_image_base64 = numpy_to_base64(heatmap_image, RESIZE_SIZE)
            # if highlighted_image is not None:
            #     highlighted_image_base64 = numpy_to_base64(highlighted_image, RESIZE_SIZE)
            visualization_end_time = time.time()
            visualization_time = (visualization_end_time - visualization_start_time) * 1000

        result = {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "boundary_image_base64": boundary_image_base64,
            "heatmap_image_base64": heatmap_image_base64,
            # "highlighted_image_base64": highlighted_image_base64
        }
        
        total_end_time = time.time() # End total timing
        total_latency = (total_end_time - total_start_time) * 1000 # Latency in milliseconds

        # Log performance metrics
        logger.info(f"Performance Metrics: Total Latency: {total_latency:.2f}ms, Preprocessing Time: {preprocess_time:.2f}ms, Inference Time: {inference_time:.2f}ms, Visualization Time: {visualization_time:.2f}ms")

        # Log Anomaly Detection Metrics
        score_map_min = np.min(score_maps)
        score_map_max = np.max(score_maps)
        score_map_mean = np.mean(score_maps)
        score_map_std = np.std(score_maps)
        logger.info(f"Anomaly Detection Metrics: Anomaly Score: {anomaly_score:.4f}, Is Anomaly: {is_anomaly}, Threshold: {ANOMALY_THRESHOLD}, Score Map Stats (Min/Max/Mean/Std): {score_map_min:.4f}/{score_map_max:.4f}/{score_map_mean:.4f}/{score_map_std:.4f}")

        return json.dumps(result)
    except Exception as e:
        error = str(e)
        logger.error(f"Error during inference: {error}")
        import traceback
        traceback.print_exc()
        return json.dumps({"error": error})



# # -------------------------------------------------------------------------------------------------------
# # score.py
# #
# # This script is designed for Azure Machine Learning real-time inference. It is converted from a FastAPI
# # application to be compatible with Azure ML's scoring service. It includes the required `init()` and
# # `run()` functions.
# #
# # - The `init()` function loads the ONNX model and sets up global variables.
# # - The `run()` function processes incoming data, performs inference, and returns predictions.
# # -------------------------------------------------------------------------------------------------------

# import json
# import os
# import numpy as np
# import onnxruntime
# from PIL import Image
# import io
# import base64
# import static.anodet as anodet

# # Global variables for the model and configuration
# sess = None
# ANOMALY_THRESHOLD = 13.0
# RESIZE_SIZE = (224, 224)

# def init():
#     """
#     This function is called when the service is initialized.
#     It loads the ONNX model and sets up the inference session.
#     """
#     global sess, ANOMALY_THRESHOLD, RESIZE_SIZE
    
#     # The model is expected to be in the same directory as this script, or in a subdirectory.
#     # Azure ML copies the model files to the same directory as the scoring script.
#     model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "padim_model.onnx")
#     model_path = "./model_output/padim_model.onnx"
#     try:
#         if os.path.exists(model_path):
#             sess = onnxruntime.InferenceSession(model_path)
#             print("ONNX model loaded successfully from:", model_path)
#         else:
#             raise FileNotFoundError(f"Model file not found at: {model_path}")
#     except Exception as e:
#         raise RuntimeError(f"Failed to load ONNX model: {e}")

# def run(raw_data):
#     """
#     This function is called for every invocation of the endpoint.
#     It handles the incoming data, performs inference, and returns the prediction.
#     """
#     global sess, ANOMALY_THRESHOLD, RESIZE_SIZE

#     if sess is None:
#         return {"error": "Model is not loaded. The service may not have initialized correctly."}

#     try:
#         # The input data is expected to be a JSON string with a base64-encoded image.
#         data = json.loads(raw_data)
#         image_base64 = data.get("image")
#         include_visualizations = data.get("include_visualizations", True)

#         if not image_base64:
#             return {"error": "No image found in the request. Please provide a base64-encoded image in the 'image' field."}

#         # Decode the base64 image
#         image_bytes = base64.b64decode(image_base64)
        
#         # Preprocess the image
#         image_np = preprocess_image_from_upload(image_bytes)

#         # Perform inference using the ONNX model
#         batch = anodet.to_batch([image_np])
#         input_name = sess.get_inputs()[0].name
#         output_names = [output.name for output in sess.get_outputs()]
        
#         outputs = sess.run(output_names, {input_name: batch})
        
#         image_scores = np.array([outputs[0]])
#         score_maps = np.array(outputs[1])

#         # Process the results
#         anomaly_score = float(image_scores[0])
#         is_anomaly = anomaly_score >= ANOMALY_THRESHOLD

#         # Initialize visualization strings
#         boundary_image_base64 = ""
#         heatmap_image_base64 = ""

#         if include_visualizations:
#             boundary_image, heatmap_image, _ = create_visualizations(image_np, score_maps, image_scores)
#             if boundary_image is not None:
#                 boundary_image_base64 = numpy_to_base64(boundary_image, RESIZE_SIZE)
#             if heatmap_image is not None:
#                 heatmap_image_base64 = numpy_to_base64(heatmap_image, RESIZE_SIZE)

#         # Prepare the response
#         result = {
#             "anomaly_score": anomaly_score,
#             "is_anomaly": is_anomaly,
#             "boundary_image_base64": boundary_image_base64,
#             "heatmap_image_base64": heatmap_image_base64
#         }

#         return result

#     except Exception as e:
#         import traceback
#         error_message = f"Prediction failed: {str(e)}"
#         traceback.print_exc()
#         return {"error": error_message, "traceback": traceback.format_exc()}



# def preprocess_image_from_upload(file_contents: bytes) -> np.ndarray:
#     """Convert uploaded file to numpy array matching detect.py preprocessing"""
#     image_pil = Image.open(io.BytesIO(file_contents)).convert("RGB")
#     image_np = np.array(image_pil)
#     return image_np

# def create_visualizations(image_np: np.ndarray, score_maps: np.ndarray, image_scores: np.ndarray) -> tuple:
#     """Create visualization images using NumPy arrays (torch-free)."""
#     try:
#         score_map_classifications = anodet.classification(score_maps, ANOMALY_THRESHOLD)
#         image_classifications = anodet.classification(image_scores, ANOMALY_THRESHOLD)
#         test_images = np.array([image_np])

#         boundary_images = anodet.visualization.framed_boundary_images(
#             test_images, score_map_classifications, image_classifications, padding=40
#         )
#         heatmap_images = anodet.visualization.heatmap_images(
#             test_images, score_maps, alpha=0.5
#         )
#         highlighted_images = anodet.visualization.highlighted_images(
#             [image_np], score_map_classifications, color=(128, 0, 128)
#         )

#         return boundary_images[0], heatmap_images[0], highlighted_images[0]

#     except Exception as e:
#         print(f"Visualization error: {e}")
#         return None, None, None

# def numpy_to_base64(image_array: np.ndarray, resize_to: tuple[int, int] = None) -> str:
#     """Convert numpy array to base64-encoded PNG, optionally resized"""
#     if image_array is None:
#         return ""
#     try:
#         if image_array.dtype != np.uint8:
#             if image_array.max() <= 1.0:
#                 image_array = (image_array * 255).astype(np.uint8)
#             else:
#                 image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
#         image = Image.fromarray(image_array)
#         if resize_to is not None:
#             image = image.resize(resize_to, Image.BILINEAR)
#         buffered = io.BytesIO()
#         image.save(buffered, format="PNG")
#         return base64.b64encode(buffered.getvalue()).decode("utf-8")
#     except Exception as e:
#         print(f"Error converting to base64: {e}")
#         return ""


