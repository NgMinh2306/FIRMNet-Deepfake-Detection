# train/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger

logger = get_logger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # class weight tensor
        self.gamma = gamma
        self.reduction = reduction
        logger.info(f"Initialized FocalLoss with gamma={gamma}, reduction={reduction}")

    def forward(self, inputs, targets):
        alpha = self.alpha.to(inputs.device) if self.alpha is not None else None
        ce_loss = F.cross_entropy(inputs, targets, weight=alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SoftFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', alpha=None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha.clone().detach()
            else:
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        logger.info(f"Initialized SoftFocalLoss with gamma={gamma}, reduction={reduction}")

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device).view(1, -1)
            focal_weight = focal_weight * alpha

        loss = -targets * focal_weight * log_probs
        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FocalLoss and SoftFocalLoss")
    parser.add_argument('--soft', action='store_true', help='Use SoftFocalLoss instead of FocalLoss')
    args = parser.parse_args()

    logger.info("Running CLI test for losses.py")

    # Fix random seed here (using when testing)
    # torch.manual_seed(42)

    # Sample input
    inputs = torch.randn(4, 2, requires_grad=True)
    targets_int = torch.tensor([0, 1, 1, 0])            # For FocalLoss
    targets_soft = F.one_hot(targets_int, num_classes=2).float()  # For SoftFocalLoss

    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)

    if args.soft:
        criterion = SoftFocalLoss(alpha=class_weights, gamma=2.0)
        loss = criterion(inputs, targets_soft)
        logger.info(f"SoftFocalLoss output: {loss.item():.4f}")
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        loss = criterion(inputs, targets_int)
        logger.info(f"FocalLoss output: {loss.item():.4f}")