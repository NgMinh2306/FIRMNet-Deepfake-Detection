# models/backbones.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger
from models.fagate import FrequencyAttentionGate
from models.fer import FrequencyEnhancedResidual

logger = get_logger(__name__)

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(in_channels, reduced_channels, 1, bias=True)
        self.act = nn.SiLU()
        self.expand = nn.Conv2d(reduced_channels, in_channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        se = self.pool(x)
        se = self.reduce(se)
        se = self.act(se)
        se = self.expand(se)
        se = self.gate(se)
        return x * se


class StochasticDepth(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1 - self.p
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x / keep_prob * binary_tensor


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio=1, se_ratio=0.25, drop_rate=0.0):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = in_ch * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU()
            ])

        layers.extend([
            nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU()
        ])

        self.block = nn.Sequential(*layers)
        self.se = SqueezeExcite(mid_ch, se_ratio)
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.stochastic_depth = StochasticDepth(drop_rate)

    def forward(self, x):
        identity = x
        out = self.block(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = self.stochastic_depth(out)
            out += identity
        return out


class FusedMBConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, drop_rate=0.0):
        super().__init__()
        self.use_residual = (in_ch == out_ch and stride == 1)

        if expand_ratio == 1:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            )
        else:
            mid_ch = in_ch * expand_ratio
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 3, stride, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU(),
                nn.Conv2d(mid_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        self.stochastic_depth = StochasticDepth(drop_rate)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = self.stochastic_depth(out)
            out += x
        return out

class FAGateBlockStack(nn.Module):
    def __init__(self, in_channels, config_list):
        """
        FAGateBlockStack: Consists of multiple stacked FrequencyAttentionGate (FAGate) layers.

        Args:
            in_channels (int): Number of input channels.
            config_list (List[str]): List of regularization methods for each layer 
                                     (zero/mask/None). The last layer will always 
                                     apply convolution after IFFT.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        total_layers = len(config_list)

        for idx, cfg in enumerate(config_list):
            self.layers.append(
                FrequencyAttentionGate(
                    in_channels=in_channels,
                    target='Channel',
                    regularization=cfg,
                    target_band='low+mid' if cfg == 'zero' else 'high',
                    mask_ratio=0.3 if cfg == 'mask' else 0.0,
                    scale=3,
                    fft_norm='ortho',
                    apply_convolution=(idx == total_layers - 1) 
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FERBlockStack(nn.Module):
    def __init__(self, in_channels, num_layers, use_groupnorm=False):
        """
        FERBlockStack: Consists of multiple stacked FrequencyEnhancedResidual layers.
        
        Args:
            in_channels (int): Number of input channels.
            num_layers (int): Number of FER layers (usually = number of layers in the previous stage).
            use_groupnorm (bool): If True, use GroupNorm in each FER.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            FrequencyEnhancedResidual(
                in_channels=in_channels,
                target='Full-image',
                fft_norm='ortho',
                use_groupnorm=use_groupnorm
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for fer in self.layers:
            x = fer(x)
        return x
    
class FIRMNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU()
        )

        self.stage1 = nn.Sequential(
            FusedMBConv(24, 24, 1, 1),
            FusedMBConv(24, 24, 1, 1)
        )

        self.stage2 = nn.Sequential(
            FusedMBConv(24, 48, 2, 4),
            FusedMBConv(48, 48, 1, 4),
            FusedMBConv(48, 48, 1, 4),
            FusedMBConv(48, 48, 1, 4)
        )
        self.fag1 = FAGateBlockStack(48, ['zero', None, None])

        self.stage3 = nn.Sequential(
            FusedMBConv(48, 64, 2, 4, 0.05),
            FusedMBConv(64, 64, 1, 4, 0.05),
            FusedMBConv(64, 64, 1, 4, 0.05),
            FusedMBConv(64, 64, 1, 4, 0.05)
        )
        self.fag2 = FAGateBlockStack(64, ['mask', None, None])

        self.stage4 = nn.Sequential(
            MBConv(64, 128, 2, 4, 0.25, 0.1),
            *[MBConv(128, 128, 1, 4, 0.25, 0.1) for _ in range(5)]
        )
        self.fer = FERBlockStack(128, 4, use_groupnorm=True)

        self.stage5 = nn.Sequential(
            MBConv(128, 160, 1, 6, 0.25, 0.15),
            *[MBConv(160, 160, 1, 6, 0.25, 0.15) for _ in range(8)]
        )

        self.stage6 = nn.Sequential(
            MBConv(160, 256, 2, 6, 0.25, 0.2),
            *[MBConv(256, 256, 1, 6, 0.25, 0.2) for _ in range(14)]
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.fag1(x)
        x = self.stage3(x)
        x = self.fag2(x)
        x = self.stage4(x)
        x = self.fer(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.head(x)
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run EfficientNetV2-S with FAGate and FER on an input image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (RGB)")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--save_logits", type=str, default=None, help="Path to save logits as .pt file")
    parser.add_argument("-t", "--target", type=str, choices=["Full-image", "Channel"], default="Full-image")
    parser.add_argument("-r", "--regularization", type=str, choices=["mask", "zero", "none"], default="none")
    parser.add_argument("-b", "--band", type=str, default="high", help="Target frequency band")
    parser.add_argument("-m", "--mask_ratio", type=float, default=0.15, help="Mask ratio (only for mask)")
    parser.add_argument("-s", "--scale", type=int, default=3, help="Frequency region scale (>2)")
    parser.add_argument("--relu", action="store_true", help="Apply ReLU after IFFT")
    parser.add_argument("--groupnorm", action="store_true", help="Use GroupNorm (default: Identity)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load image
    image = Image.open(args.image).convert('RGB')
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run model
    model = FIRMNet(num_classes=args.num_classes).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
    logger.info(f"Input shape: {input_tensor.shape}, Logits shape: {logits.shape}")
    logger.info(f"Model config: num_classes={args.num_classes}")

    logger.info(f"Model output (logits): {logits}")

    if args.save_logits:
        torch.save(logits.cpu(), args.save_logits)
        logger.info(f"Logits saved to {args.save_logits}")
