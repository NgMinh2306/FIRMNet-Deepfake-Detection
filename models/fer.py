# models/fer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from PIL import Image
import torchvision.transforms as T

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger

logger = get_logger(__name__)


class FrequencyEnhancedResidual(nn.Module):
    """
    Frequency-Enhanced Residual (FER): Enhances features via frequency domain.

    Args:
        in_channels (int): Number of input channels.
        target (str): 'Full-image' or 'Channel' FFT domain.
        scale (int): Frequency scale (defines low/mid/high regions).
        fft_norm (str): Normalization for FFT ('ortho', 'forward', 'backward').
        use_groupnorm (bool): Whether to use GroupNorm instead of Identity.
    """

    def __init__(self, in_channels, target='Full-image', scale=3, fft_norm='ortho', use_groupnorm=False):
        super().__init__()
        assert target in ['Full-image', 'Channel']
        self.target = target
        self.scale = scale
        self.fft_norm = fft_norm

        NormLayer = lambda channels: nn.GroupNorm(8, channels) if use_groupnorm else nn.Identity()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            NormLayer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            NormLayer(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W, "Input must be square."

        x = x.float() if x.dtype == torch.float16 else x

        if self.target == 'Full-image':
            fft = torch.fft.fft2(x, dim=(-2, -1), norm=self.fft_norm)
            fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        else:
            fft = torch.fft.fft(x, dim=1, norm=self.fft_norm)
            fft_shifted = torch.fft.fftshift(fft, dim=1)

        amp = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)

        enhanced_amp = self.cnn(amp)
        enhanced_fft = enhanced_amp * torch.exp(1j * phase)

        if self.target == 'Full-image':
            enhanced_fft = torch.fft.ifftshift(enhanced_fft, dim=(-2, -1))
            res = torch.fft.ifft2(enhanced_fft, dim=(-2, -1), norm=self.fft_norm).real
        else:
            enhanced_fft = torch.fft.ifftshift(enhanced_fft, dim=1)
            res = torch.fft.ifft(enhanced_fft, dim=1, norm=self.fft_norm).real

        return x + res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply FrequencyEnhancedResidual (FER) to an input image.")
    parser.add_argument('--image', type=str, required=True, help='Path to input image (RGB)')
    parser.add_argument('-t', '--target', type=str, choices=['Full-image', 'Channel'], default='Full-image')
    parser.add_argument('-s', '--scale', type=int, default=3, help='Frequency region scale (>2)')
    parser.add_argument('--groupnorm', action='store_true', help='Use GroupNorm (default: Identity)')
    args = parser.parse_args()

    # Load and preprocess image
    image = Image.open(args.image).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]

    fer = FrequencyEnhancedResidual(
        in_channels=3,
        target=args.target,
        scale=args.scale,
        use_groupnorm=args.groupnorm
    )

    output = fer(img_tensor)
    output_img = output.squeeze(0).clamp(0, 1)

    logger.info(f"FER applied. Output shape: {output.shape}")
