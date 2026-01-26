"""
Drift feature extraction utilities
Compatible with both single images (score.py) and batches (train.py)
"""
import numpy as np
import pandas as pd
from datetime import datetime


def extract_drift_features(data, scores, maps=None, include_metadata=True):
    """
    Optimized vectorized extraction that works for:
    1. Single 3D images (score.py): data shape (H, W, C)
    2. 4D Batches (train.py): data shape (B, H, W, C) or (B, C, H, W)

    Args:
        data: Image data (3D or 4D numpy array)
        scores: Anomaly scores (scalar or 1D array)
        maps: Score maps (optional, 3D or 4D)
        include_metadata: Whether to include timestamp and image dimensions

    Returns:
        DataFrame with drift features
    """
    # Determine if single image or batch
    is_batch = data.ndim == 4

    if not is_batch:
        # Single image: add batch dimension
        data = np.expand_dims(data, axis=0)
        scores = np.atleast_1d(scores)
        if maps is not None:
            maps = np.expand_dims(maps, axis=0)

    # Now data is always 4D: (B, H, W, C) or (B, C, H, W)
    batch_size = data.shape[0]

    # Detect channel order (C, H, W) vs (H, W, C)
    # Assume channels are the smallest dimension and < 10
    if data.shape[1] < 10:  # (B, C, H, W)
        spatial_axes = (2, 3)  # H, W
        height_idx, width_idx, channel_idx = 2, 3, 1
    else:  # (B, H, W, C)
        spatial_axes = (1, 2)  # H, W
        height_idx, width_idx, channel_idx = 1, 2, 3

    # Calculate pixel statistics over spatial dimensions
    pixel_means = np.mean(data, axis=spatial_axes + (channel_idx,))
    pixel_stds = np.std(data, axis=spatial_axes + (channel_idx,))
    pixel_mins = np.min(data, axis=spatial_axes + (channel_idx,))
    pixel_maxs = np.max(data, axis=spatial_axes + (channel_idx,))

    # Ensure scores is correct shape
    scores = np.atleast_1d(scores).flatten()
    if len(scores) != batch_size:
        scores = np.full(batch_size, scores[0])

    # Base row data
    row_data = {
        "pixel_mean": pixel_means.flatten(),
        "pixel_std": pixel_stds.flatten(),
        "pixel_min": pixel_mins.flatten(),
        "pixel_max": pixel_maxs.flatten(),
        "anomaly_score": scores,
    }

    # Add image dimensions if metadata requested
    if include_metadata:
        row_data["img_height"] = np.full(batch_size, data.shape[height_idx])
        row_data["img_width"] = np.full(batch_size, data.shape[width_idx])
        row_data["img_channels"] = np.full(batch_size, data.shape[channel_idx])
        row_data["timestamp"] = [datetime.now().isoformat()] * batch_size

    # Calculate score map statistics if available
    if maps is not None:
        # maps shape: (B, H, W) for batch
        map_spatial_axes = tuple(range(maps.ndim - 2, maps.ndim))

        row_data["score_map_max"] = np.max(maps, axis=map_spatial_axes).flatten()
        row_data["score_map_mean"] = np.mean(maps, axis=map_spatial_axes).flatten()
        row_data["score_map_min"] = np.min(maps, axis=map_spatial_axes).flatten()
        row_data["score_map_std"] = np.std(maps, axis=map_spatial_axes).flatten()

    return pd.DataFrame(row_data)

