#HSIFeatureNet


import torch
import torch.nn as nn
import torch.nn.functional as F

class HSIFeatureNet(nn.Module):
    def __init__(self, num_spectral_bands, out_features=64):
        super().__init__()
        self.input_conv = nn.Conv2d(num_spectral_bands, 64, 3, padding=1)
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

    def forward(self, hsi):
        """
        Entrée :
            hsi: (B, C, H, W) — image hyperspectrale avec C bandes
        Sortie :
            features:     (B, D, H, W)
            saliency_map: (B, 1, H, W)
        """
        x = F.relu(self.input_conv(hsi))
        x = self.block1(x)
        x = self.block2(x)
        features = self.output_conv(x)

        # Carte de saillance via norme L2 par pixel
        saliency = torch.norm(features, dim=1, keepdim=True)  # (B, 1, H, W)

        return features, saliency
