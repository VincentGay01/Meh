import torch
import torch.nn as nn
import torch.nn.functional as F

class RTIFeatureNet(nn.Module):
    def __init__(self, num_rti_views, out_features=64):
        super().__init__()
        self.input_conv = nn.Conv2d(num_rti_views, 64, 3, padding=1)  # ⚠️ plus de "+3"
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )
        self.output_conv = nn.Conv2d(64, out_features, 1)

    def forward(self, rti_stack):
        """
        rti_stack: (B, N, H, W)
        """
        x = F.relu(self.input_conv(rti_stack))
        x = self.block1(x)
        x = self.block2(x)
        features = self.output_conv(x)  # (B, D, H, W)
        return features
