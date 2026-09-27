"""Model architecture and preprocessing shared by training and the API.

Both train_model.py and api/main.py import from here so the exact same
architecture and image transform are used at train time and at inference
time -- a mismatch between the two is a classic source of silent accuracy
loss in production.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

# Alphabetical order, fixed once and reused everywhere (dataset folder names,
# model output indices, API response labels). Changing this order after a
# model has been trained would silently mislabel every prediction.
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

IMAGE_SIZE = 224

# ImageNet mean/std -- required because the backbone was pretrained on
# ImageNet-normalized inputs, even though our images are grayscale.
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]


def build_transform() -> transforms.Compose:
    """Preprocessing applied to every image, train and inference alike."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
        ]
    )


class DefectClassifier(nn.Module):
    """MobileNetV2 feature extractor (frozen) + a small trained head.

    MobileNetV2 over ResNet18: its ImageNet backbone is ~13MB vs ~45MB,
    which keeps the trained artifact under the repo's commit-friendly size
    budget -- see the README's engineering-decisions section.
    """

    def __init__(self, num_classes: int = len(CLASS_NAMES), pretrained: bool = False):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen-backbone features only, used to cache embeddings for fast
        head-only training."""
        with torch.no_grad():
            x = self.features(x)
            return self.pool(x).flatten(1)


def load_model(weights_path: Path) -> DefectClassifier:
    """Build the architecture with random weights, then load our own
    trained state dict. No internet access needed at inference time --
    the ImageNet-pretrained backbone is only downloaded once, during
    training (see train_model.py)."""
    model = DefectClassifier(pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
