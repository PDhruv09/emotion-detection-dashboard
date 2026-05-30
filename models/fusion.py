import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights, EfficientNet_B0_Weights, ResNet18_Weights


class FusionEmotionNet(nn.Module):
    def __init__(
        self,
        landmark_size,
        num_classes=7,
        image_backbone="efficientnet_b0",
        landmark_hidden=256,
        fusion_hidden=512,
        dropout=0.35,
        pretrained=True,
    ):
        super().__init__()
        self.image_backbone_name = image_backbone

        if image_backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            image_feature_size = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        elif image_backbone == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            image_feature_size = backbone.classifier[2].in_features
            backbone.classifier = nn.Sequential(
                backbone.classifier[0],
                backbone.classifier[1],
                nn.Flatten(1),
            )
        elif image_backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            image_feature_size = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported image backbone: {image_backbone}")

        self.image_backbone = backbone
        self.landmark_encoder = nn.Sequential(
            nn.Linear(landmark_size, landmark_hidden),
            nn.LayerNorm(landmark_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(landmark_hidden, landmark_hidden),
            nn.LayerNorm(landmark_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(image_feature_size + landmark_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, num_classes),
        )

    def set_image_backbone_trainable(self, trainable):
        for parameter in self.image_backbone.parameters():
            parameter.requires_grad = trainable

    def forward(self, images, landmarks):
        image_features = self.image_backbone(images)
        if image_features.ndim > 2:
            image_features = torch.flatten(image_features, 1)
        landmark_features = self.landmark_encoder(landmarks)
        features = torch.cat([image_features, landmark_features], dim=1)
        return self.classifier(features)
