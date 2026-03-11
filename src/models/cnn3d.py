"""
Plain 3D CNN encoder-decoder for brain tumor segmentation.

NO SKIP CONNECTIONS — this is the key difference from U-Net.
Used as a baseline to demonstrate the value of skip connections.
Same channel progression and layer counts as the U-Net for fair comparison.
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Double convolution block: Conv3d → InstanceNorm → LeakyReLU × 2"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNN3D(nn.Module):
    """
    Plain 3D CNN encoder-decoder WITHOUT skip connections.

    Architecture mirrors U-Net structure (same encoder/decoder depth,
    same channel counts) but deliberately omits skip connections.
    This isolates the exact contribution of skip connections.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: list = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock3D(prev_ch, ch, dropout))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_ch = ch

        # --- Bottleneck ---
        bottleneck_ch = channels[-1] * 2
        self.bottleneck = ConvBlock3D(channels[-1], bottleneck_ch, dropout)

        # --- Decoder ---
        # NO SKIP CONNECTIONS: input to decoder block is just upsampled features
        # (In U-Net, it would be upsampled + skip = 2× channels)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_ch = bottleneck_ch
        for ch in reversed(channels):
            self.upconvs.append(
                nn.ConvTranspose3d(prev_ch, ch, kernel_size=2, stride=2)
            )
            # Input is ONLY the upsampled features (no skip concatenation)
            self.decoders.append(ConvBlock3D(ch, ch, dropout))
            prev_ch = ch

        # --- Final output ---
        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path (save nothing — no skip connections)
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path (no skip connections)
        for upconv, decoder in zip(self.upconvs, self.decoders):
            x = upconv(x)
            x = decoder(x)

        return self.final_conv(x)


def get_cnn_model(cfg: dict) -> nn.Module:
    """Build a plain 3D CNN model from configuration."""
    model_cfg = cfg["model"]

    model = CNN3D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        channels=model_cfg["channels"],
        dropout=model_cfg.get("dropout", 0.0),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 3D CNN (no skip connections) created")
    print(f"[Model]   Total parameters:     {total_params:,}")
    print(f"[Model]   Trainable parameters: {trainable_params:,}")
    print(f"[Model]   Channels: {model_cfg['channels']}")
    print(f"[Model]   In/Out: {model_cfg['in_channels']}/{model_cfg['out_channels']}")

    return model
