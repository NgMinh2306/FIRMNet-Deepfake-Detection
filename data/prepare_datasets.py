# data/prepare_datasets.py

import os
import argparse
from collections import Counter
from torchvision import datasets
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger

logger = get_logger(__name__)

def split_dataset_by_class(data_dir, val_ratio=0.15, random_state=42, logger=None):
    """
    Split an ImageFolder dataset (with 'FAKE' and 'REAL' folders) into class-balanced train/val sets.

    Args:
        data_dir (str): Path to the root folder containing 'FAKE' and 'REAL' subfolders.
        val_ratio (float): Ratio of validation data (e.g., 0.15 = 15% for validation).
        random_state (int): Seed for reproducibility.
        logger: Optional logger object from logger.py

    Returns:
        train_indices (List[int]), val_indices (List[int])
    """
    dataset = datasets.ImageFolder(root=data_dir)
    targets = dataset.targets
    class_to_idx = dataset.class_to_idx

    if "FAKE" not in class_to_idx or "REAL" not in class_to_idx:
        raise ValueError("The dataset folder must contain 'FAKE' and 'REAL' subfolders.")

    fake_idx = class_to_idx['FAKE']
    real_idx = class_to_idx['REAL']

    fake_indices = [i for i, label in enumerate(targets) if label == fake_idx]
    real_indices = [i for i, label in enumerate(targets) if label == real_idx]

    fake_train, fake_val = train_test_split(fake_indices, test_size=val_ratio, random_state=random_state)
    real_train, real_val = train_test_split(real_indices, test_size=val_ratio, random_state=random_state)

    train_indices = fake_train + real_train
    val_indices = fake_val + real_val

    if logger:
        logger.info("Dataset statistics:")
        logger.info(f" - FAKE total: {len(fake_indices)} → train: {len(fake_train)} | val: {len(fake_val)}")
        logger.info(f" - REAL total: {len(real_indices)} → train: {len(real_train)} | val: {len(real_val)}")
        logger.info(f" - Total train: {len(train_indices)}, Total val: {len(val_indices)}")

    return train_indices, val_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split FAKE/REAL dataset into train/val subsets.")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to folder containing 'FAKE/' and 'REAL/' subfolders.")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save log files")

    args = parser.parse_args()
    logger = get_logger(name="prepare_dataset", log_dir=args.log_dir)

    logger.info("Starting dataset split...")
    split_dataset_by_class(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        logger=logger
    )
    logger.info("Dataset split completed.")
