import sys, os
import pandas as pd
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

from utils import extract_drift_features
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
    parser.add_argument('--run_name', type=str, default='anomaV', help='MLflow run name (auto-generated if not provided)')
    parser.add_argument('--registered_model_name', type=str, default='PadimONNX')

    # Evaluation
    parser.add_argument('--evaluate_model', action='store_true', default=False, help='Evaluate after training')
    parser.add_argument('--threshold', type=float, default=13.0, help='Anomaly detection threshold')

    # Drift monitoring
    parser.add_argument('--generate_baseline', action='store_true', default=True, help='Generate baseline reference data for drift monitoring')
    parser.add_argument('--baseline_output_path', type=str, default='outputs/baseline_reference.csv', help='Path to save baseline reference dataset')

    # Logging
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')

    return parser, parser.parse_args()

def evaluate_model(model, test_dataloader, device, threshold):
    """
    Evaluate trained model and return metrics.

    This function ONLY calculates evaluation metrics (AUC, accuracy, etc.)
    """
    logger.info("Starting model evaluation...")
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch, _, image_classification, _ in test_dataloader:
            # Move batch to device and run inference
            batch_device = batch.to(device)
            image_scores, score_maps = model.predict(batch_device)

            # Convert to numpy for metric calculation
            image_scores_np = image_scores.cpu().numpy()

            # Collect predictions and ground truth
            all_scores.extend(image_scores_np)
            all_labels.extend(image_classification.numpy())

    # Calculate evaluation metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

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


def generate_baseline_reference(model, dataloader, device, output_path, threshold, dataset_type="test"):
    """
    Generate baseline reference dataset for drift monitoring.

    This function is specifically for creating the reference dataset
    that will be used to compare against production data.

    Args:
        model: Trained PaDiM model
        dataloader: DataLoader with data to use as baseline
        device: torch device
        output_path: Path to save baseline CSV
        threshold: Anomaly detection threshold
        dataset_type: 'train' or 'test' - what type of data this is

    Returns:
        DataFrame with baseline reference data
    """
    logger.info(f"Generating baseline reference dataset from {dataset_type} data...")
    model.eval()
    all_drift_dfs = []

    with torch.no_grad():
        for batch_idx, (batch, _, image_classification, _) in enumerate(dataloader):
            # Move batch to device and run inference
            batch_device = batch.to(device)
            image_scores, score_maps = model.predict(batch_device)

            # Convert to numpy for drift feature extraction
            batch_np = batch.cpu().numpy()  # Shape: (B, C, H, W) or (B, H, W, C)
            image_scores_np = image_scores.cpu().numpy()  # Shape: (B,)
            score_maps_np = score_maps.cpu().numpy()  # Shape: (B, H, W)

            try:
                # Extract drift features for entire batch
                batch_df = extract_drift_features(
                    data=batch_np,
                    scores=image_scores_np,
                    maps=score_maps_np,
                    include_metadata=True
                )

                # Add ground truth labels and metadata
                batch_df['is_anomaly'] = image_classification.numpy().astype(int)
                batch_df['threshold'] = threshold
                batch_df['dataset_type'] = dataset_type
                batch_df['batch_idx'] = batch_idx

                all_drift_dfs.append(batch_df)

                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Processed {batch_idx + 1} batches for baseline reference")

            except Exception as e:
                logger.warning(f"Failed to extract drift features for batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()

    # Concatenate all batches
    if not all_drift_dfs:
        logger.error("No drift features collected. Cannot generate baseline reference.")
        return None

    df_reference = pd.concat(all_drift_dfs, ignore_index=True)

    # Add summary statistics
    logger.info(f"Baseline reference dataset generated:")
    logger.info(f"  Total samples: {len(df_reference)}")
    logger.info(f"  Normal samples: {(df_reference['is_anomaly'] == 0).sum()}")
    logger.info(f"  Anomaly samples: {(df_reference['is_anomaly'] == 1).sum()}")
    logger.info(f"  Mean anomaly score: {df_reference['anomaly_score'].mean():.4f}")
    logger.info(f"  Std anomaly score: {df_reference['anomaly_score'].std():.4f}")

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_reference.to_csv(output_path, index=False)
    logger.info(f"Baseline reference saved to: {output_path}")

    return df_reference


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

    run_name = config.run_name or f"padim_{config.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load datasets
    train_dataset = AnomaVision.AnodetDataset(os.path.join(DATASET_PATH, "train/good"))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    logger.info(f"Training images: {len(train_dataset)}")

    test_dataloader = None
    if config.evaluate_model or config.generate_baseline:
        try:
            test_path = os.path.dirname(DATASET_PATH)
            class_name = os.path.basename(DATASET_PATH)
            test_dataset = AnomaVision.MVTecDataset(test_path, class_name=class_name, is_train=False)
            test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)
            logger.info(f"Test images: {len(test_dataset)}")
        except Exception as e:
            logger.warning(f"Could not load test dataset: {e}")
            config.evaluate_model = False
            config.generate_baseline = False

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
                "generate_baseline": config.generate_baseline,
            })
            log_environment(device)

            # ====================================================================
            # STEP 1: TRAIN MODEL
            # ====================================================================
            logger.info("=" * 80)
            logger.info("STEP 1: Training PaDiM model...")
            logger.info("=" * 80)

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
            logger.info(f"PyTorch model saved to: {pytorch_model_path}")

            # ====================================================================
            # STEP 2: EXPORT TO ONNX
            # ====================================================================
            logger.info("=" * 80)
            logger.info("STEP 2: Exporting to ONNX...")
            logger.info("=" * 80)

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

            # ====================================================================
            # STEP 3: EVALUATE MODEL (Optional)
            # ====================================================================
            if config.evaluate_model and test_dataloader:
                logger.info("=" * 80)
                logger.info("STEP 3: Evaluating model performance...")
                logger.info("=" * 80)

                metrics = evaluate_model(padim, test_dataloader, device, config.threshold)
                mlflow.log_metrics(metrics)

            # ====================================================================
            # STEP 4: GENERATE BASELINE REFERENCE (Separate from evaluation)
            # ====================================================================
            if config.generate_baseline and train_dataloader:
                logger.info("=" * 80)
                logger.info("STEP 4: Generating baseline reference for drift monitoring...")
                logger.info("=" * 80)

                baseline_df = generate_baseline_reference(
                    model=padim,
                    dataloader=train_dataloader,
                    device=device,
                    output_path=config.baseline_output_path,
                    threshold=config.threshold,
                    dataset_type="test"
                )

                if baseline_df is not None:
                    # Log baseline statistics to MLflow
                    mlflow.log_metrics({
                        "baseline_samples": len(baseline_df),
                        "baseline_normal_samples": int((baseline_df['is_anomaly'] == 0).sum()),
                        "baseline_anomaly_samples": int((baseline_df['is_anomaly'] == 1).sum()),
                        "baseline_mean_score": float(baseline_df['anomaly_score'].mean()),
                        "baseline_std_score": float(baseline_df['anomaly_score'].std()),
                    })

                    # Upload baseline to MLflow
                    mlflow.log_artifact(config.baseline_output_path, "baseline")
                    logger.info(f"✓ Baseline reference uploaded to MLflow")

            # ====================================================================
            # STEP 5: SAVE METADATA
            # ====================================================================
            logger.info("=" * 80)
            logger.info("STEP 5: Saving model metadata...")
            logger.info("=" * 80)

            model_info = {
                "model_type": "PaDiM",
                "backbone": config.backbone,
                "input_shape": [1, config.input_channels, config.input_height, config.input_width],
                "framework": "PyTorch",
                "export_format": "ONNX",
                "training_time_seconds": training_time,
                "threshold": config.threshold,
                "created_at": datetime.now().isoformat(),
                "baseline_generated": config.generate_baseline,
            }

            model_info_path = os.path.join(MODEL_DATA_PATH, "model_info.json")
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)

            mlflow.log_artifact(model_info_path, "metadata")
            mlflow.log_artifact(onnx_model_path, "models")

            # Save effective configuration
            save_args_to_yaml(config, os.path.join(MODEL_DATA_PATH, "config.yml"))
            mlflow.log_artifact(os.path.join(MODEL_DATA_PATH, "config.yml"), "metadata")

            # ====================================================================
            # SUMMARY
            # ====================================================================
            logger.info("=" * 80)
            logger.info("✓ Training pipeline completed successfully!")
            logger.info("=" * 80)
            logger.info(f"MLflow run ID: {run.info.run_id}")
            logger.info(f"PyTorch model: {pytorch_model_path}")
            logger.info(f"ONNX model: {onnx_model_path}")
            if config.generate_baseline:
                logger.info(f"Baseline reference: {config.baseline_output_path}")
            logger.info("=" * 80)

        except Exception as e:
            mlflow.log_param("training_status", "failed")
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
