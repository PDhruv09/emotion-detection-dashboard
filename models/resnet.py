# models/resnet.py
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionResNet, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adjust for grayscale images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Add dropout before final classification
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
