import os
import argparse
import json
from PIL import Image

def validate_mvtec_structure(dataset_path):
    required_subdirs = ['train', 'test', 'ground_truth']
    issues = {}

    if not os.path.exists(dataset_path):
        issues['dataset'] = f"Dataset path '{dataset_path}' does not exist."
        return issues

    for sub in required_subdirs:
        full_path = os.path.join(dataset_path, sub)
        if not os.path.isdir(full_path):
            issues[sub] = f"Missing required folder: {sub}"

    return issues

def check_images_validity(folder_path):
    issues = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    issues.append(f"{img_path}: {str(e)}")
    return issues

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data_validation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Validation report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True, help="Path to MVTec dataset")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    args = parser.parse_args()

    print("🔍 Validating MVTec dataset structure...")
    issues = validate_mvtec_structure(args.input_data)

    print("🖼️ Checking image integrity...")
    corrupt_images = check_images_validity(args.input_data)
    if corrupt_images:
        issues['corrupt_images'] = corrupt_images

    result = {"status": "passed"} if not issues else {"status": "failed", "issues": issues}
    save_results(result, args.output_dir)

    if result["status"] == "failed":
        print("❌ Data validation failed.")
        exit(1)
    else:
        print("✅ Data validation passed.")
