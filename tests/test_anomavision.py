import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# from padim import Padim
# import src.AnomaVision.anodet as anodet




from src.AnomaVision.anodet import Padim


@pytest.fixture
def dummy_dataloader():
    images = torch.rand(8, 3, 224, 224)  # 8 RGB images
    labels = torch.zeros(8, dtype=torch.long)
    masks = torch.zeros(8, 224, 224, dtype=torch.uint8)
    dataset = TensorDataset(images, labels, masks)
    return DataLoader(dataset, batch_size=2)


@pytest.mark.parametrize("backbone, expected_total_channels", [
    ("resnet18", sum([64, 128, 256, 512])),       # [0,1,2,3]
    ("wide_resnet50", sum([255, 512, 1024, 2048]))
])
def test_channel_indices_dimension(backbone, expected_total_channels):
    model = Padim(backbone=backbone, device=torch.device("cpu"), layer_indices=[0, 1, 2, 3], feat_dim=1000)
    assert model.channel_indices.numel() <= expected_total_channels
    assert model.channel_indices.ndim == 1


@pytest.mark.parametrize("backbone", ["resnet18", "wide_resnet50"])
def test_fit_and_predict_all_layers(dummy_dataloader, backbone):
    model = Padim(backbone=backbone, device=torch.device("cpu"), layer_indices=[0, 1, 2, 3], feat_dim=100)
    model.fit(dummy_dataloader)

    assert model.mahalanobisDistance is not None
    assert model.mahalanobisDistance._mean_flat.shape[1] == 100

    batch = next(iter(dummy_dataloader))[0]
    image_scores, score_map = model.predict(batch)

    assert image_scores.shape[0] == batch.shape[0]
    assert score_map.shape[1:] == batch.shape[2:]


@pytest.mark.parametrize("backbone", ["resnet18", "wide_resnet50"])
def test_forward_and_evaluate(dummy_dataloader, backbone):
    model = Padim(backbone=backbone, device=torch.device("cpu"), layer_indices=[0, 1, 2, 3], feat_dim=128)
    model.fit(dummy_dataloader)

    batch = next(iter(dummy_dataloader))[0]
    image_scores, score_map = model(batch)
    assert image_scores.ndim == 1
    assert score_map.ndim == 3

    images, labels, masks, scores, maps = model.evaluate(dummy_dataloader)
    assert images.shape[0] == labels.shape[0] == scores.shape[0] == 8
    assert masks.ndim == 1
    assert maps.ndim == 1
