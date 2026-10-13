import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet201


CLASS_NAMES = [
    "Bacterial_Leaf_Blight",
    "Healthy",
    "Septoria_Brown_Spot",
    "Vein_Necrosis",
]

NUM_CLASSES = len(CLASS_NAMES)


class RoboCropSoybeanV3(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.backbone = densenet201(weights=None)

        feature_dim = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        features = self.backbone.features(x)

        features = torch.relu(features)

        features = F.adaptive_avg_pool2d(
            features,
            (1, 1)
        )

        features = torch.flatten(
            features,
            1
        )

        return self.classifier(features)