
import json
import base64
import os
from score import init, run  # assumes your ONNX code is in 'score.py'

def image_to_base64(file_path: str) -> str:
    """
    Reads an image file from disk and returns its base64 representation
    with the appropriate MIME type prefix.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".webp": "image/webp"
    }

    if ext not in mime_map:
        raise ValueError(f"Unsupported image format: {ext}")

    with open(file_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")

    return f"{encoded_str}"


# Step 1: Initialize the model
init()

# Step 2: Prepare input data
image_path = "000.png"  # <-- Replace this with your actual image path
include_visualizations = False
threshold = 13.0  # Optional: change threshold if needed

# Convert image to base64
try:
    image_base64 = image_to_base64(image_path)

    # Create input dictionary
    input_data = {
        "image_base64": image_base64,
        "include_visualizations": include_visualizations,
        "threshold": threshold
    }

    # Convert to raw JSON string for `run`
    raw_data = json.dumps(input_data)

    # Step 3: Run inference
    for i in range(2):
        result = run(raw_data)

    # Step 4: Print the result nicely formatted
    # print(json.dumps(json.loads(result), indent=2))

except Exception as e:
    print(f"Error: {e}")