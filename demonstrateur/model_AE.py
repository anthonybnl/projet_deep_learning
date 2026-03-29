from pathlib import Path

from torch import nn
import torch
from torchvision.models import resnet18, ResNet18_Weights

MODEL_PATH = Path.cwd().parent / "models" / "chestmnist_autoencoder.pth"

class ChestMNIST_autoencoder(nn.Module):
    def __init__(self, latent_dim=128, img_size=64):
        super().__init__()

        self.img_size = img_size

        # encodeur : 64x64 vers un espace latent a 128 dimensions
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (64, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (128, 8, 8)
        )

        final_img_size = img_size // 8  # 3 max pool donc 2*2*2 = 8

        # features vers espace latent
        self.encoder_feat_to_latent = nn.Linear(
            128 * final_img_size * final_img_size, latent_dim
        )

        # décodeur : espace latent vers features
        self.decoder_latent_to_feat = nn.Linear(
            latent_dim, 128 * final_img_size * final_img_size
        )

        # Décodeur : espace latent 128 vers image 64*64
        self.decoder_upconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # (1, 64, 64)
            nn.Sigmoid(),  # sortie ∈ [0, 1]
        )

    def encode(self, x):
        h = self.encoder_conv(x) # (B, 128, 8, 8)
        h = h.view(h.size(0), -1) # flatten des données sauf batch (B, 128*8*8)
        return self.encoder_feat_to_latent(h) # projectiotest_loadern vers espace latent (B, latent)

    def decode(self, z):
        h = self.decoder_latent_to_feat(z) # espace latent vers features (B, 128*8*8)
        h = h.view(h.size(0), 128, 8, 8) # reshape vers (B, 128, 8, 8)
        return self.decoder_upconv(h) # upconv vers image (B, 1, 64, 64)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

def get_model_ae(device = 'cpu'):
    model = ChestMNIST_autoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model