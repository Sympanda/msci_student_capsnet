import torch.nn as nn
import torch.nn.functional as F

# ---- Feature Extractor (Custom CNN Example) ---- #
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super(FeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_layers(x)  # Output shape: (batch_size, feature_dim, H, W)
