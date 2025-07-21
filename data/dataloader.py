# data/dataloader.py

import argparse
import json
import os
import yaml
from types import SimpleNamespace

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from data.transforms import (
    get_train_transforms,
    get_valid_transforms,
    AlbumentationsTransform,
    apply_shortcut_flags,
)
from data.prepare_datasets import split_dataset_by_class
from logger import get_logger


def get_dataloaders(data_dir, batch_size=32, val_ratio=0.15, num_workers=2, transform_opt=None, logger=None):
    """
    Create PyTorch dataloaders for training and validation with Albumentations transforms.

    Args:
        data_dir (str): Path to dataset folder containing 'FAKE/' and 'REAL/' subfolders.
        batch_size (int): Batch size for training and validation.
        val_ratio (float): Ratio of validation data.
        num_workers (int): Number of workers for DataLoader.
        transform_opt: Namespace object containing transform params for train.
        logger: Logger object for logging output.

    Returns:
        train_loader (DataLoader), val_loader (DataLoader)
    """
    logger = logger or get_logger(name="dataloader", log_dir="logs", log_filename="main.log")
    
    train_indices, val_indices = split_dataset_by_class(
        data_dir=data_dir, val_ratio=val_ratio, logger=logger
    )

    train_transform = AlbumentationsTransform(get_train_transforms(transform_opt))
    val_transform = AlbumentationsTransform(get_valid_transforms())

    full_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    full_val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if logger:
        logger.info(f"DataLoader ready: train={len(train_dataset)}, val={len(val_dataset)}")

    return train_loader, val_loader


def load_config(config_path):
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported config format. Use .json or .yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create DataLoaders with transform configs")

    # Config loading
    parser.add_argument('--config', type=str, default=None, help="Path to .json or .yaml config file")

    # Shortcuts
    parser.add_argument('--augment_strong', action='store_true')
    parser.add_argument('--color_jitter', action='store_true')

    # Logging + data paths
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset with FAKE/REAL subfolders")
    parser.add_argument('--log_dir', type=str, default="logs")

    # Data split & loader
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)

    # Transforms (same như transforms.py)
    parser.add_argument('--resize_max', type=int, default=256)
    parser.add_argument('--crop_height', type=int, default=224)
    parser.add_argument('--crop_width', type=int, default=224)
    parser.add_argument('--hflip_prob', type=float, default=0.5)

    parser.add_argument('--affine_trans', type=float, default=0.05)
    parser.add_argument('--affine_scale', type=float, default=0.05)
    parser.add_argument('--affine_rotate', type=int, default=5)
    parser.add_argument('--affine_p', type=float, default=0.3)

    parser.add_argument('--blur_limit', type=int, default=3)
    parser.add_argument('--blur_p', type=float, default=0.0)

    parser.add_argument('--noise_std_min', type=float, default=10/255)
    parser.add_argument('--noise_std_max', type=float, default=50/255)
    parser.add_argument('--noise_p', type=float, default=0.0)

    parser.add_argument('--brightness_contrast_p', type=float, default=0.0)
    parser.add_argument('--hsv_p', type=float, default=0.0)
    parser.add_argument('--rgbshift_p', type=float, default=0.0)
    parser.add_argument('--clahe_p', type=float, default=0.0)

    parser.add_argument('--downscale_range', type=float, nargs=2, default=[0.25, 0.75])
    parser.add_argument('--downscale_p', type=float, default=0.0)
    parser.add_argument('--jpeg_quality', type=int, nargs=2, default=[30, 80])
    parser.add_argument('--jpeg_p', type=float, default=0.0)

    parser.add_argument('--coarse_num_holes', type=int, nargs=2, default=[3, 6])
    parser.add_argument('--coarse_h_range', type=float, nargs=2, default=[0.05, 0.15])
    parser.add_argument('--coarse_w_range', type=float, nargs=2, default=[0.05, 0.15])
    parser.add_argument('--coarse_p', type=float, default=0.0)

    parser.add_argument('--use_imagenet_norm', action='store_true')

    args = parser.parse_args()
    logger = get_logger(name="dataloader", log_dir=args.log_dir)

    logger.info("Starting DataLoader setup...")

    # Load config if provided
    if args.config:
        logger.info(f"Loading config from {args.config}")
        loaded_cfg = load_config(args.config)
        for k, v in loaded_cfg.items():
            setattr(args, k, v)

    # Apply shortcut flags
    apply_shortcut_flags(args)

    # Log final transform settings
    logger.info("Final transform config:")
    logger.info(json.dumps(vars(args), indent=2))

    # Build dataloaders
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        transform_opt=args,
        logger=logger
    )