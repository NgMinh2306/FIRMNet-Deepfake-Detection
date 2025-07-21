# train/optimizer.py

import torch
import argparse
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
    OneCycleLR, ReduceLROnPlateau, LambdaLR
)

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger

logger = get_logger(__name__)


def get_optimizer_adam(model, lr=1e-4, weight_decay=1e-6, betas=(0.9, 0.999), eps=1e-7):
    """
    Returns Adam optimizer with specified parameters.
    """
    logger.info(f"Creating Adam optimizer: lr={lr}, weight_decay={weight_decay}, betas={betas}, eps={eps}")
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )


def get_optimizer_sgd(model, lr=1e-3, momentum=0.9, weight_decay=0):
    """
    Returns SGD optimizer with default parameters.
    """
    logger.info(f"Creating SGD optimizer: lr={lr}, momentum={momentum}, weight_decay={weight_decay}")
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )


def get_optimizer_adamw(model, lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8):
    """
    Returns AdamW optimizer with default parameters.
    """
    logger.info(f"Creating AdamW optimizer: lr={lr}, weight_decay={weight_decay}, betas={betas}, eps={eps}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )


def choose_optimizer(model, optimizer_type='adam', **kwargs):
    """
    Choose between 'adam', 'sgd', or 'adamw'.
    kwargs will be passed to the corresponding optimizer function.
    """
    optimizer_type = optimizer_type.lower()
    if optimizer_type == 'adam':
        return get_optimizer_adam(model, **kwargs)
    elif optimizer_type == 'sgd':
        return get_optimizer_sgd(model, **kwargs)
    elif optimizer_type == 'adamw':
        return get_optimizer_adamw(model, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


def get_scheduler(optimizer, num_epochs, train_loader_len, scheduler_type='cosine', warmup_type='linear',
                  warmup_epochs=0, scheduler_params=None):
    scheduler = None
    warmup_scheduler = None
    scheduler_params = scheduler_params or {}

    logger.info(f"Setting scheduler: {scheduler_type}, Warmup: {warmup_type}, Warmup_epochs: {warmup_epochs}")

    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, **scheduler_params)
    elif scheduler_type in ['cosinerestart', 'cosinewarmrestart']:
        scheduler = CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    elif scheduler_type == 'onecycle':
        total_steps = num_epochs * train_loader_len
        scheduler = OneCycleLR(optimizer, total_steps=total_steps, **scheduler_params)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
    else:
        logger.warning("Unknown scheduler_type specified.")

    if warmup_type and warmup_epochs > 0:
        def linear_warmup(epoch):
            return min(1.0, epoch / warmup_epochs)

        def cosine_warmup(epoch):
            from math import cos, pi
            return 0.5 * (1 + cos(pi * (1 - epoch / warmup_epochs)))

        warmup_lambda = linear_warmup if warmup_type == 'linear' else cosine_warmup
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        logger.info(f"Using {warmup_type} warmup for {warmup_epochs} epochs.")

    return scheduler, warmup_scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test optimizer and scheduler")
    parser.add_argument('--epochs', type=int, default=10, help="Total number of epochs")
    parser.add_argument('--sched', type=str, default='cosine', help="Scheduler type")
    parser.add_argument('--warmup', type=str, default='linear', help="Warmup type")
    parser.add_argument('--warmup_epochs', type=int, default=3, help="Number of warmup epochs")
    parser.add_argument('--opt', type=str, default='adam', help="Optimizer type: adam | sgd | adamw")
    args = parser.parse_args()

    logger.info("Running CLI test for optimizer.py")

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)

    model = DummyModel()
    optimizer = choose_optimizer(model, optimizer_type=args.opt)
    scheduler, warmup = get_scheduler(optimizer, num_epochs=args.epochs, train_loader_len=100,
                                      scheduler_type=args.sched,
                                      warmup_type=args.warmup,
                                      warmup_epochs=args.warmup_epochs)
    
    logger.info("Test completed.")
