"""
CNN model for urban sound classification from mel-spectrograms.

Defines the UrbanSoundCNN architecture (4 conv blocks, ~390K parameters)
and a helper function to load a saved checkpoint.
"""

import torch
import torch.nn as nn

from src.dataset import NUM_CLASSES


class UrbanSoundCNN(nn.Module):
    """4-block CNN for 10-class urban sound classification from mel-spectrograms.

    Input shape: (batch, 1, 128, 173) — single-channel log-mel spectrogram.

    Architecture:
        Block 1: Conv2d(1→32)   + BN + ReLU + MaxPool2d(2)
        Block 2: Conv2d(32→64)  + BN + ReLU + MaxPool2d(2)
        Block 3: Conv2d(64→128) + BN + ReLU + MaxPool2d(2)
        Block 4: Conv2d(128→256)+ BN + ReLU + MaxPool2d(2)
        Classifier: AdaptiveAvgPool2d(1) + Flatten + Dropout(0.3) + Linear(256, n_classes)

    Total parameters: ~390K.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(1,   32),
            conv_block(32,  64),
            conv_block(64,  128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_cnn_model(checkpoint_path, device="cpu"):
    """Load a trained UrbanSoundCNN from a checkpoint file.

    Handles state dicts saved from plain training as well as those saved
    after torch.compile(), which prefixes every key with ``_orig_mod.``.

    Args:
        checkpoint_path: Path to a ``.pt`` file containing a state dict.
        device: Device string (e.g. ``'cpu'``, ``'cuda'``).

    Returns:
        UrbanSoundCNN instance in eval mode, on the requested device.
    """
    model = UrbanSoundCNN()
    state_dict = torch.load(checkpoint_path, map_location=device)

    # torch.compile() wraps parameters under '_orig_mod.' — strip the prefix
    # so the checkpoint can be loaded into an uncompiled model.
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
