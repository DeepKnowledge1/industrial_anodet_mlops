from src.model.anodet.utils import to_batch, standard_image_transform

import os
import torch
import cv2



DATASET_PATH = os.path.realpath(r"D:\01-DATA\bottle")




import onnxruntime as ort

import cv2


# --- Load ONNX model
MODEL_DATA_PATH = "."
onnx_model_path = os.path.join(MODEL_DATA_PATH, "padim_model.onnx")
session = ort.InferenceSession(
    onnx_model_path,
    providers=["CPUExecutionProvider"]
)


# --- Input and output names
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]
print("Input:", input_name)
print("Outputs:", output_names)

# --- Load and preprocess input image
def preprocess_image(image_path, input_size=(224, 224)):
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    batch = to_batch([image], standard_image_transform, torch.device('cpu'))
    
    return batch.numpy()

# --- Run inference

paths = [
    os.path.join(DATASET_PATH, "test/broken_large/000.png"),
    os.path.join(DATASET_PATH, "test/broken_small/000.png"),
    os.path.join(DATASET_PATH, "test/contamination/000.png"),
    os.path.join(DATASET_PATH, "test/good/000.png"),
    os.path.join(DATASET_PATH, "test/good/001.png"),
]

for image_path in paths:
    input_tensor = preprocess_image(image_path)
    # print(input_tensor.shape)
    
    outputs = session.run(output_names, {input_name: input_tensor})

    # --- Postprocess
    image_scores = outputs[0]  # shape: (1,)
    score_map = outputs[1]     # shape: (1, H, W)

    print("Image Score:", image_scores[0])


