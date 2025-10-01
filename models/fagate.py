# models/fagate.py

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
from models.frequency_utils import global_frequency_masking, frequency_suppression

logger = get_logger(__name__)


class FrequencyAttentionGate(nn.Module):
    """
    Frequency Attention Gate (FAGate): Attention mechanism in frequency domain.
    Applies FFT to extract frequency representation, generates attention from real part, 
    then modulates original input.

    Args:
        in_channels (int): Number of input channels.
        target (str): 'Full-image' or 'Channel'.
        regularization (str): None, 'mask', or 'zero'.
        target_band (str): Frequency region to apply masking/suppression.
        mask_ratio (float): Ratio of masking if using 'mask'.
        fft_norm (str): 'ortho', 'forward', or 'backward'.
        apply_convolution (bool): If True, apply Conv3x3+ReLU and Conv1x1+ReLU after IFFT.
        scale (int): Frequency region scale factor (>2).
    """

    def __init__(self, in_channels, target='Full-image', regularization=None, target_band='all',
                 mask_ratio=0.15, fft_norm='ortho', apply_convolution=False, scale=3):
        super().__init__()
        assert target in ['Full-image', 'Channel']
        assert regularization in [None, 'mask', 'zero']

        self.in_channels = in_channels
        self.target = target
        self.regularization = regularization
        self.target_band = target_band
        self.mask_ratio = mask_ratio
        self.scale = scale
        self.fft_norm = fft_norm
        self.apply_convolution = apply_convolution

        self.attn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # conv stack sau IFFT (chỉ dùng nếu apply_convolution=True)
        self.post_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        ) if apply_convolution else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W, "Input must be square."

        dim_size = H if self.target == 'Full-image' else C
        x = x.float() if x.dtype == torch.float16 else x

        if self.target == 'Full-image':
            fft = torch.fft.fft2(x, dim=(-2, -1), norm=self.fft_norm)
            fft = torch.fft.fftshift(fft, dim=(-2, -1))
        else:
            fft = torch.fft.fft(x, dim=1, norm=self.fft_norm)
            fft = torch.fft.fftshift(fft, dim=1)

        # Regularization
        if self.training and self.regularization == 'mask':
            fft = global_frequency_masking(
                fft, mask_ratio=self.mask_ratio, target_band=self.target_band,
                target=self.target, scale=self.scale, dim_size=dim_size
            )
        elif self.regularization == 'zero':
            fft = frequency_suppression(
                fft, target_band=self.target_band, target=self.target, scale=self.scale
            )

        # Attention from real part
        attn = torch.sigmoid(self.attn_conv(fft.real))
        fft = fft * attn

        # IFFT
        if self.target == 'Full-image':
            fft = torch.fft.ifftshift(fft, dim=(-2, -1))
            out = torch.fft.ifft2(fft, dim=(-2, -1), norm=self.fft_norm).real
        else:
            fft = torch.fft.ifftshift(fft, dim=1)
            out = torch.fft.ifft(fft, dim=1, norm=self.fft_norm).real

        # Dropout-style scaling at inference
        if not self.training and self.regularization == 'mask':
            out = out * (1 - self.mask_ratio)

        # Conv3x3 + Conv1x1 stack nếu được yêu cầu
        out = self.post_conv(out)
        return out



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply FrequencyAttentionGate (FAGate) to an input image.")
    parser.add_argument('--image', type=str, required=True, help='Path to input image (RGB)')
    parser.add_argument('-t', '--target', type=str, choices=['Full-image', 'Channel'], default='Full-image')
    parser.add_argument('-r', '--regularization', type=str, choices=['mask', 'zero', 'none'], default='none')
    parser.add_argument('-b', '--band', type=str, default='high', help='Target frequency band')
    parser.add_argument('-m', '--mask_ratio', type=float, default=0.15, help='Mask ratio (only for mask)')
    parser.add_argument('-s', '--scale', type=int, default=3, help='Frequency region scale (>2)')
    parser.add_argument('--apply_conv', action='store_true', help='Apply Conv3x3+ReLU and Conv1x1+ReLU after IFFT')
    args = parser.parse_args()

    # Load and preprocess image
    image = Image.open(args.image).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]

    regularization = None if args.regularization == 'none' else args.regularization

    fagate = FrequencyAttentionGate(
        in_channels=3,
        target=args.target,
        regularization=regularization,
        target_band=args.band,
        mask_ratio=args.mask_ratio,
        scale=args.scale,
        apply_convolution=args.apply_conv
    )

    output = fagate(img_tensor)
    output_img = output.squeeze(0).clamp(0, 1)

    logger.info(f"FAGate applied. Output shape: {output.shape}")
