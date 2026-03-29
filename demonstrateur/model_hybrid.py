from pathlib import Path
from torch import nn
import torch
from torchvision.models import resnet18, ResNet18_Weights

MODEL_PATH = Path.cwd().parent / "models" / "cnn_chestmnist_hybride.pth"

class Hybride_ChestMNIST(nn.Module):
    def __init__(self, num_labels=14, freeze_cnn=True):
        super().__init__()

        # resnet18 pré-entraîné.
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # on prends toutes les couches sauf les 2 dernières (avgpool + fc)
        # donc on garde l'extraction de features mais pas le classifieur de ResNet
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # → (B, 512, H', W')

        # on gèle les poids du CNN si freeze_cnn est True
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # tokenisation : transformer necessite une dimension 256.
        self.proj = nn.Linear(512, 256)

        # Transformer encodeur multi-têtes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification multi-label
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, x):
        feat = self.cnn(x)  # (B, 512, H', W')
        feat_flattened = feat.flatten(2)  # (B, 512, H'*W')
        tokens = feat_flattened.permute(0, 2, 1)  # (B, H*W, 512)
        tokens = self.proj(tokens)  # (B, H*W, 256)
        out = self.transformer(tokens)  # (B, H*W, 256)
        out = out.mean(dim=1)  # (B, 1, 256) global average pooling
        return self.classifier(out)  # (B, 14) → sigmoid pour multi-label

def get_model_hybrid(device = 'cpu'):
    model = Hybride_ChestMNIST()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model