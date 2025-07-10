
import os
import AnomaVision.anodet as anodet
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from AnomaVision.export import export_onnx
import argparse
import mlflow
import mlflow.onnx
import onnx
from datetime import datetime
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import logging


os.environ.pop("MLFLOW_RUN_ID", None)  # Remove Azure-injected env var

from logger import setup_logging

setup_logging()
# Now use the logger
logger = logging.getLogger("industrial-mlops")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PaDiM model for anomaly detection with MLflow tracking.")

    parser.add_argument('--dataset_path', default="D:/01-DATA/bottle", type=str, required=False,
                        help='Path to the dataset folder containing "train/good" images.')

    parser.add_argument('--model_data_path', type=str, default='./distributions/',
                        help='Directory to save model distributions and ONNX file.')
    parser.add_argument('--output_model', type=str, default='model_output', help='Output folder for model export')
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'wide_resnet50'], default='resnet18',
                        help='Backbone network to use for feature extraction.')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size used during training and inference.')

    # parser.add_argument('--output_model', type=str, default='padim_model.pt',
    #                     help='Filename to save the PT model.')

    parser.add_argument('--layer_indices', nargs='+', type=int, default=[0],
                        help='List of layer indices to extract features from. Default: [0].')

    parser.add_argument('--feat_dim', type=int, default=50,
                        help='Number of random feature dimensions to keep.')

    # MLflow specific arguments
    parser.add_argument('--mlflow_tracking_uri', type=str, default='file:./mlruns',
                        help='MLflow tracking URI.')

    parser.add_argument('--mlflow_experiment_name', type=str, default='padim_anomaly_detection',
                        help='MLflow experiment name.')

    parser.add_argument('--run_name', type=str, default=None,
                        help='MLflow run name. If not provided, will be auto-generated.')

    parser.add_argument('--registered_model_name', type=str, default='PadimONNX',
                        help='Name for the registered model in MLflow Model Registry.')

    # Evaluation arguments
    parser.add_argument('--test_dataset_path', type=str, default="D:/01-DATA/bottle",
                        help='Path to test dataset for evaluation. If not provided, will use dataset_path/test.')

    parser.add_argument('--evaluate_model', action='store_true',
                        help='Whether to evaluate the model after training.')

    return parser.parse_args()

def evaluate_model(model, test_dataloader, device):
    """
    Evaluate the trained model and return metrics.
    
    Args:
        model: Trained PaDiM model
        test_dataloader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info("Starting model evaluation...")
    
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch, labels, _ in test_dataloader:
            batch = batch.to(device)
            
            # Get predictions
            image_scores, _ = model.predict(batch)
            
            # Convert to numpy
            scores = image_scores.cpu().numpy()
            labels_np = labels.numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels_np)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    try:
        auc_score = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc_score = 0.0
        logger.warning("Could not calculate AUC score - possibly only one class in test set")
    
    # Calculate precision-recall AUC
    try:
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        pr_auc = auc(recall, precision)
    except ValueError:
        pr_auc = 0.0
        logger.warning("Could not calculate PR-AUC")
    
    # Calculate accuracy with threshold (using threshold from detect.py)
    threshold = 13.0
    predictions = (all_scores > threshold).astype(int)
    accuracy = np.mean(predictions == all_labels)
    
    metrics = {
        'auc_score': float(auc_score),
        'pr_auc': float(pr_auc),
        'accuracy': float(accuracy),
        'mean_anomaly_score': float(np.mean(all_scores)),
        'std_anomaly_score': float(np.std(all_scores)),
        'threshold': threshold
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

def create_sample_input_output(model, device, input_shape=(1, 3, 224, 224)):
    """
    Create sample input and output for MLflow model signature.
    
    Args:
        model: Trained PaDiM model
        device: Device to run the model on
        input_shape: Shape of input tensor
        
    Returns:
        tuple: (sample_input, sample_output) as numpy arrays
    """
    model.eval()
    with torch.no_grad():
        sample_input = torch.randn(*input_shape).to(device)
        sample_output = model(sample_input)
        
        # Convert to numpy
        sample_input_np = sample_input.cpu().numpy()
        
        if isinstance(sample_output, tuple):
            # If output is tuple (image_scores, score_maps)
            sample_output_np = {
                'anomaly_score': sample_output[0].cpu().numpy(),
                'anomaly_map': sample_output[1].cpu().numpy()
            }
        else:
            sample_output_np = sample_output.cpu().numpy()
    
    return sample_input_np, sample_output_np

def main(args):
    # Set up paths
    DATASET_PATH = os.path.realpath(args.dataset_path)
    MODEL_DATA_PATH = os.path.realpath(args.model_data_path)
    os.makedirs(MODEL_DATA_PATH, exist_ok=True)

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set up MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(args.mlflow_experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(args.mlflow_experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        logger.warning(f"Could not create/get experiment: {e}")
        experiment_id = None
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"padim_{args.backbone}_{timestamp}"

    # Load training dataset
    train_dataset = anodet.AnodetDataset(os.path.join(DATASET_PATH, "train/good"))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    logger.info(f"Number of training images: {len(train_dataset)}")

    # Load test dataset if evaluation is requested
    test_dataloader = None
    if args.evaluate_model:
        
        test_path = os.path.dirname(DATASET_PATH)
        class_name = os.path.basename(DATASET_PATH)
        
        # test_path = args.test_dataset_path or DATASET_PATH
        try:
            # Try to load as MVTec dataset for evaluation
            test_dataset = anodet.MVTecDataset(test_path, class_name=class_name, is_train=False)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
            logger.info(f"Number of test images: {len(test_dataset)}")
        except Exception as e:
            logger.warning(f"Could not load test dataset: {e}")
            args.evaluate_model = False

    # Start MLflow run
    if experiment_id:
        mlflow_run_ctx = mlflow.start_run(run_name=args.run_name, experiment_id=experiment_id)
    else:
        mlflow_run_ctx = mlflow.start_run(run_name=args.run_name)

    with mlflow_run_ctx:
        try:
            # Log parameters
            mlflow.log_param("backbone", args.backbone)
            mlflow.log_param("batch_size", args.batch_size)
            mlflow.log_param("layer_indices", str(args.layer_indices))
            mlflow.log_param("feat_dim", args.feat_dim)
            mlflow.log_param("dataset_path", args.dataset_path)
            mlflow.log_param("device", str(device))
            mlflow.log_param("num_training_images", len(train_dataset))
            
            # Log system info
            mlflow.log_param("torch_version", torch.__version__)
            mlflow.log_param("cuda_available", torch.cuda.is_available())
            if torch.cuda.is_available():
                mlflow.log_param("cuda_device_name", torch.cuda.get_device_name())

            # Initialize model
            logger.info("Initializing PaDiM model...")
            padim = anodet.Padim(
                backbone=args.backbone,
                device=device,
                layer_indices=args.layer_indices,
                feat_dim=args.feat_dim
            )

            # Train model
            logger.info("Training PaDiM model...")
            start_time = datetime.now()
            padim.fit(train_dataloader)
            training_time = (datetime.now() - start_time).total_seconds()
            
            mlflow.log_metric("training_time_seconds", training_time)
            logger.info(f"Training completed in {training_time:.2f} seconds")

            # Save PyTorch model
            # pytorch_model_path = os.path.join(MODEL_DATA_PATH, args.output_model)
            # torch.save(padim, pytorch_model_path)
            
            # Log PyTorch model as artifact
            # mlflow.log_artifact(pytorch_model_path, "models")

            # Export to ONNX
            logger.info("Exporting model to ONNX...")
            onnx_model_path = os.path.join(MODEL_DATA_PATH, "padim_model.onnx")
            export_onnx(padim, onnx_model_path)

            # Save a copy of the ONNX model to AzureML's output folder
            try:
                os.makedirs(args.output_model, exist_ok=True)
                azure_output_path = os.path.join(args.output_model, "padim_model.onnx")
                import shutil
                shutil.copyfile(onnx_model_path, azure_output_path)
                logger.info(f"Saved model to AzureML output folder: {azure_output_path}")
            except Exception as e:
                logger.warning(f"Could not copy model to AzureML output: {e}")
            
            # Load ONNX model for MLflow logging
            onnx_model = onnx.load(onnx_model_path)
            
            # Create model signature
            try:
                sample_input, sample_output = create_sample_input_output(padim, device)
                signature = mlflow.models.infer_signature(sample_input, sample_output)
            except Exception as e:
                logger.warning(f"Could not create model signature: {e}")
                signature = None

            # Log ONNX model to MLflow
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="onnx_model",
                registered_model_name=args.registered_model_name,
                signature=signature
            )

            # Evaluate model if requested
            if args.evaluate_model and test_dataloader is not None:
                metrics = evaluate_model(padim, test_dataloader, device)
                
                # Log evaluation metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

            # Log model artifacts
            mlflow.log_artifact(onnx_model_path, "models")
            
            # Create and log model info
            model_info = {
                "model_type": "PaDiM",
                "backbone": args.backbone,
                "input_shape": [1, 3, 224, 224],
                "output_shape": "variable",
                "framework": "PyTorch",
                "export_format": "ONNX",
                "training_dataset": args.dataset_path,
                "training_time_seconds": training_time,
                "created_at": datetime.now().isoformat()
            }
            
            # Save model info as JSON
            model_info_path = os.path.join(MODEL_DATA_PATH, "model_info.json")
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            mlflow.log_artifact(model_info_path, "metadata")

            # Log success
            mlflow.log_param("training_status", "success")
            logger.info("Training and MLflow logging completed successfully!")
            
            # Print MLflow run info
            run = mlflow.active_run()
            if run:
                logger.info(f"MLflow run ID: {run.info.run_id}")
                logger.info(f"MLflow run URL: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

        except Exception as e:
            # Log failure
            mlflow.log_param("training_status", "failed")
            mlflow.log_param("error_message", str(e))
            logger.error(f"Training failed: {e}")
            raise

if __name__ == "__main__":
    args = parse_args()
    main(args)

