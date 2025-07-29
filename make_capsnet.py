import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from feature_extractor import FeatureExtractor
from primary_caps import PrimaryCapsules
from attention import SelfAttentionCapsules
from beta_loss import BetaScheduler, CyclicalBetaScheduler, kl_divergence, loss_function
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Based on a config file, make a fully functional capsnet with an
# encoder and decoder

import torch
import torch.nn as nn

class CCN_Decoder(nn.Module):
    """
    Decoder for unsupervised CapsNet/VAE model.

    Projects a latent vector into a 3x128x128 (RGB) or 1x128x128 (grayscale) image.
    Starts by reshaping latent space into a small feature map, then upsamples with ConvTranspose2d.
    """

    def __init__(self, latent_dim, output_size=(128, 128), output_channels=3, initial_channels=256):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_size = output_size
        self.output_channels = output_channels
        self.initial_channels = initial_channels
        self.feature_map_size = 8  # Starting spatial size for reshaped latent projection

        # Fully connected projection: [B, latent_dim] → [B, C × H × W]
        self.fc = nn.Linear(latent_dim, initial_channels * self.feature_map_size * self.feature_map_size)

        # Upsampling network: 8×8 → 128×128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(initial_channels, 128, kernel_size=4, stride=2, padding=1),  # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # 64→128
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        """
        z: latent vector of shape [B, latent_dim]
        Returns: decoded image [B, C, H, W]
        """
        B = z.size(0)
        x = self.fc(z)  # [B, C*H*W]
        x = x.view(B, self.initial_channels, self.feature_map_size, self.feature_map_size)  # [B, C, 8, 8]
        x = self.decoder(x)  # [B, output_channels, 128, 128]
        return x
    
class BetaCapsNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        feature_dim=64,
        num_primary_capsules=8,
        primary_caps_dim=16,
        output_caps_dim=16,
        decoder=None,
        image_size=128,
    ):
        super().__init__()

        self.feature_extractor = FeatureExtractor(in_channels, feature_dim)

        self.primary_caps = PrimaryCapsules(
            in_channels=feature_dim,
            num_capsules=num_primary_capsules,
            capsule_dim=primary_caps_dim,
            kernel_size=7,
            stride=2,
        )

        # Compute output spatial shape of PrimaryCaps
        kernel_size = 7
        stride = 2
        H_out = (image_size - kernel_size) // stride + 1
        W_out = (image_size - kernel_size) // stride + 1
        num_input_capsules = num_primary_capsules * H_out * W_out

        self.attention_caps = SelfAttentionCapsules(
            num_input_capsules=num_input_capsules,
            num_output_capsules=1,
            input_dim=primary_caps_dim,
            output_dim=output_caps_dim,
            attention_heads=8,
        )

        self.decoder = decoder or CCN_Decoder(
            latent_dim=output_caps_dim,
            output_size=(image_size, image_size),
            output_channels=in_channels,
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x, return_latent=False):
        features = self.feature_extractor(x)
        primary_caps = self.primary_caps(features)
        mean, logvar = self.attention_caps(primary_caps)

        z = self.reparameterize(mean, logvar)
        recon = self.decoder(z)

        if return_latent:
            return recon, z, mean, logvar
        return recon, mean, logvar

    def encode(self, x):
        features = self.feature_extractor(x)
        primary_caps = self.primary_caps(features)
        mean, logvar = self.attention_caps(primary_caps)
        return self.reparameterize(mean, logvar)

    def decode(self, z):
        return self.decoder(z)
    
def build_model_from_config(config):
    """
    Build a BetaCapsNet model based on a config dictionary.
    """
    return BetaCapsNet(
        in_channels=config["input_channels"],
        feature_dim=config.get("feature_dim", 64),
        num_primary_capsules=config["num_primary_capsules"],
        primary_caps_dim=config["capsule_dim"],
        output_caps_dim=config["latent_dim"],
        decoder=CCN_Decoder(
            latent_dim=config["latent_dim"],
            output_size=(config["image_size"], config["image_size"]),
            output_channels=config["input_channels"]
        ),
        image_size=config["image_size"]
    )

if __name__ == "__main__":
    # Example test config
    config = {
        "input_channels": 3,
        "image_size": 128,
        "latent_dim": 16,
        "capsule_dim": 32,
        "num_primary_capsules": 12,
        "feature_dim": 128
    }

    # Build model
    model = build_model_from_config(config)
    print(model)

    # Dummy input
    x = torch.randn(4, config["input_channels"], config["image_size"], config["image_size"])
    recon, z, mean, logvar = model(x, return_latent=True)

    print("Input shape:", x.shape)
    print("Reconstruction shape:", recon.shape)
    print("Latent shape:", z.shape)
