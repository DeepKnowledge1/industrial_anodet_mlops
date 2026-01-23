import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "AnomaVision"))
import AnomaVision.anomavision as AnomaVision

import numpy as np
import torch
from torch.utils.data import DataLoader
from AnomaVision.export import ModelExporter
import argparse
import mlflow
import mlflow.onnx
import onnx
from datetime import datetime
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from easydict import EasyDict as edict

from AnomaVision.anomavision.utils import get_logger, setup_logging
from AnomaVision.anomavision.config import load_config
from AnomaVision.anomavision.utils import merge_config, save_args_to_yaml

setup_logging(enabled=True, log_level="INFO", log_to_file=False)
logger = get_logger("AnomaVision.train")
os.environ.pop("MLFLOW_RUN_ID", None)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PaDiM model for anomaly detection with MLflow tracking.")

    # Meta
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config.yml/.json')

    # Dataset paths
    parser.add_argument('--dataset_path', default="D:/01-DATA/bottle", type=str, help='Path to dataset folder with train/good images')
    parser.add_argument('--test_dataset_path', type=str, default=None, help='Path to test dataset (defaults to dataset_path/test)')

    # Model paths
    parser.add_argument('--model_data_path', type=str, default='./distributions/', help='Directory for distributions and ONNX')
    parser.add_argument('--output_model', type=str, default='model_padim.pt', help='PyTorch model filename')

    # Model config
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'wide_resnet50'], default='resnet18')
    parser.add_argument('--layer_indices', nargs='+', type=int, default=[0], help='Layer indices for features')
    parser.add_argument('--feat_dim', type=int, default=10, help='Random feature dimensions')
    parser.add_argument('--batch_size', type=int, default=2)

    # Input shape
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels (e.g., 3 for RGB)')
    parser.add_argument('--input_height', type=int, default=224, help='Input image height')
    parser.add_argument('--input_width', type=int, default=224, help='Input image width')

    # ONNX export
    parser.add_argument('--onnx_output_name', type=str, default='padim_model.onnx', help='ONNX model filename')
    parser.add_argument('--dynamic_batch', action='store_true', default=False, help='Enable dynamic batch size for ONNX')

    # MLflow
    parser.add_argument('--mlflow_tracking_uri', type=str, default='file:./mlruns')
    parser.add_argument('--mlflow_experiment_name', type=str, default='padim_anomaly_detection')
    parser.add_argument('--run_name', type=str, default=None, help='MLflow run name (auto-generated if not provided)')
    parser.add_argument('--registered_model_name', type=str, default='PadimONNX')

    # Evaluation
    parser.add_argument('--evaluate_model', action='store_true', default=False, help='Evaluate after training')
    parser.add_argument('--threshold', type=float, default=13.0, help='Anomaly detection threshold')

    # Logging
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')

    return parser, parser.parse_args()

def evaluate_model(model, test_dataloader, device, threshold):
    """Evaluate trained model and return metrics."""
    logger.info("Starting evaluation...")
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch, labels, _ in test_dataloader:
            image_scores, _ = model.predict(batch.to(device))
            all_scores.extend(image_scores.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Calculate metrics with error handling
    metrics = {'threshold': threshold}

    try:
        metrics['auc_score'] = float(roc_auc_score(all_labels, all_scores))
    except ValueError:
        metrics['auc_score'] = 0.0
        logger.warning("Could not calculate AUC - single class in test set")

    try:
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        metrics['pr_auc'] = float(auc(recall, precision))
    except ValueError:
        metrics['pr_auc'] = 0.0
        logger.warning("Could not calculate PR-AUC")

    predictions = (all_scores > threshold).astype(int)
    metrics['accuracy'] = float(np.mean(predictions == all_labels))
    metrics['mean_anomaly_score'] = float(np.mean(all_scores))
    metrics['std_anomaly_score'] = float(np.std(all_scores))

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def log_environment(device):
    """Log system and environment parameters."""
    mlflow.log_params({
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device)
    })
    if torch.cuda.is_available():
        mlflow.log_param("cuda_device_name", torch.cuda.get_device_name())


def export_to_onnx(model_path, output_dir, config, logger):
    """Export PyTorch model to ONNX format."""
    logger.info("Exporting to ONNX...")

    export_config = edict({
        'input_shape': [1, config.input_channels, config.input_height, config.input_width],
        'onnx_output_name': config.onnx_output_name,
        'dynamic_batch': config.dynamic_batch,
    })

    exporter = ModelExporter(str(model_path), str(output_dir), logger, device="cpu")
    onnx_path = exporter.export_onnx(
        input_shape=export_config.input_shape,
        output_name=export_config.onnx_output_name,
        dynamic_batch=export_config.dynamic_batch,
    )

    return onnx_path


def main():
    parser, args = parse_args()

    # Load config file
    if os.path.isfile(args.config):
        logger.error(f"Config file not found: {args.config}")
        cfg = load_config(args.config)
    else:
        cfg = {}

    # Merge config with CLI args (CLI args override config)
    config = edict(merge_config(args, cfg))

    # Setup logging with config
    setup_logging(enabled=True, log_level=config.log_level, log_to_file=False)
    logger = get_logger("AnomaVision.train")

    # Validate required parameters
    if not config.dataset_path:
        logger.error("dataset_path is required (via --dataset_path or config file)")
        sys.exit(1)

    # Setup paths
    DATASET_PATH = os.path.realpath(config.dataset_path)
    MODEL_DATA_PATH = os.path.realpath(config.model_data_path)
    os.makedirs(MODEL_DATA_PATH, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Setup MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    run_name = f"padim_{config.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load datasets
    train_dataset = AnomaVision.AnodetDataset(os.path.join(DATASET_PATH, "train/good"))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    logger.info(f"Training images: {len(train_dataset)}")

    test_dataloader = None
    if config.evaluate_model:
        try:
            test_path = os.path.dirname(DATASET_PATH)
            class_name = os.path.basename(DATASET_PATH)
            test_dataset = AnomaVision.MVTecDataset(test_path, class_name=class_name, is_train=False)
            test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)
            logger.info(f"Test images: {len(test_dataset)}")
        except Exception as e:
            logger.warning(f"Could not load test dataset: {e}")
            config.evaluate_model = False

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        try:
            # Log parameters
            mlflow.log_params({
                "backbone": config.backbone,
                "batch_size": config.batch_size,
                "layer_indices": str(config.layer_indices),
                "feat_dim": config.feat_dim,
                "dataset_path": config.dataset_path,
                "num_training_images": len(train_dataset),
                "input_shape": f"[1, {config.input_channels}, {config.input_height}, {config.input_width}]",
            })
            log_environment(device)

            # Initialize and train model
            logger.info("Training PaDiM model...")
            padim = AnomaVision.Padim(
                backbone=config.backbone,
                device=device,
                layer_indices=config.layer_indices,
                feat_dim=config.feat_dim
            )

            start_time = datetime.now()
            padim.fit(train_dataloader)
            training_time = (datetime.now() - start_time).total_seconds()
            mlflow.log_metric("training_time_seconds", training_time)
            logger.info(f"Training completed in {training_time:.2f}s")

            # Save PyTorch model
            pytorch_model_path = os.path.join(MODEL_DATA_PATH, config.output_model)
            torch.save(padim, pytorch_model_path)

            # Export to ONNX
            onnx_model_path = export_to_onnx(pytorch_model_path, MODEL_DATA_PATH, config, logger)

            # Log ONNX model to MLflow
            onnx_model = onnx.load(onnx_model_path)

            # Create signature
            signature = None
            try:
                sample_input = torch.randn(1, config.input_channels, config.input_height, config.input_width).to(device)
                with torch.no_grad():
                    sample_output = padim(sample_input)
                signature = mlflow.models.infer_signature(
                    sample_input.cpu().numpy(),
                    {'anomaly_score': sample_output[0].cpu().numpy(), 'anomaly_map': sample_output[1].cpu().numpy()}
                )
            except Exception as e:
                logger.warning(f"Could not create signature: {e}")

            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="onnx_model",
                registered_model_name=config.registered_model_name,
                signature=signature
            )

            # Evaluate if requested
            if config.evaluate_model and test_dataloader:
                metrics = evaluate_model(padim, test_dataloader, device, config.threshold)
                mlflow.log_metrics(metrics)

            # Save and log metadata
            model_info = {
                "model_type": "PaDiM",
                "backbone": config.backbone,
                "input_shape": [1, config.input_channels, config.input_height, config.input_width],
                "framework": "PyTorch",
                "export_format": "ONNX",
                "training_time_seconds": training_time,
                "created_at": datetime.now().isoformat()
            }

            model_info_path = os.path.join(MODEL_DATA_PATH, "model_info.json")
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)

            mlflow.log_artifact(model_info_path, "metadata")
            mlflow.log_artifact(onnx_model_path, "models")

            # Save effective configuration
            save_args_to_yaml(config, os.path.join(MODEL_DATA_PATH, "config.yml"))
            mlflow.log_artifact(os.path.join(MODEL_DATA_PATH, "config.yml"), "metadata")

            logger.info(f"✓ Training completed successfully!")
            logger.info(f"MLflow run ID: {run.info.run_id}")

        except Exception as e:
            mlflow.log_param("training_status", "failed")
            logger.error(f"Training failed: {e}")
            raise


if __name__ == "__main__":
    main()
