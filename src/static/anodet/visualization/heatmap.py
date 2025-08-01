import cv2
import numpy as np
from .utils import normalize_patch_scores, blend_image
from typing import Union, Optional


def heatmap_images(images: np.ndarray,
                   list_of_patch_scores: np.ndarray,
                   masks: Optional[np.ndarray] = None,
                   min_v: Optional[float] = None,
                   max_v: Optional[float] = None,
                   alpha: float = 0.6) -> np.ndarray:
    """
    Takes array of images and patch_scores to create heatmaps on the images.

    Args:
        images: The images to draw heatmaps on.
        list_of_patch_scores: The values to use to generate colormap.
        masks: array of masks on where to draw heatmap on images.
        min_v: min value for normalization
        max_v: max value for normalization
        alpha: The opacity of the colormap

    Returns:
        heatmaps: a array of heatmaps.

    """
    heatmaps = []
    images = (images).copy()
    list_of_patch_scores = (list_of_patch_scores).copy()

    if isinstance(masks, (np.ndarray)):
        masks = (masks).copy()

    norm_patch_scores = normalize_patch_scores(
        list_of_patch_scores,
        min_v=min_v,
        max_v=max_v
    )

    for i, score in enumerate(norm_patch_scores):
        mask = masks[i] if isinstance(masks, (np.ndarray)) else None
        image_heatmap = heatmap_image(images[i], score, mask=mask, alpha=alpha)
        heatmaps.append(image_heatmap)

    return np.array(heatmaps)


def heatmap_image(image: np.ndarray,
                  patch_scores: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  min_v: Optional[float] = None,
                  max_v: Optional[float] = None,
                  alpha: float = 0.6) -> np.ndarray:
    """
    draws a heatmap over a image using patch_scores to
    indicate areas of interest.

    Args:
        image: image to draw the colormap on.
        patch_scores: patch scores or normalized ones
        mask: mask where to draw heatmap on image.
        min_v: min value for normalization
        max_v: max value for normalization
        alpha: Opacity on the colormap

    Returns:
        heatmap: Combination of image and colormap.


    """
    image = (image).copy()
    patch_scores = (patch_scores).copy()

    if isinstance(mask, (np.ndarray)):
        mask = (mask).copy()
        mask = np.logical_not(mask).astype(np.uint8)

    if min_v and max_v:
        patch_scores = normalize_patch_scores(
            patch_scores,
            min_v=min_v,
            max_v=max_v
        )

    patch_scores = (1 - patch_scores) * 255
    patch_scores = patch_scores.astype(np.uint8)
    color_map = cv2.applyColorMap(patch_scores, colormap=cv2.COLORMAP_JET)
    heatmap = blend_image(image, color_map, alpha=alpha, mask=mask)

    return heatmap
