# train/callbacks.py

import os
import torch
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger

logger = get_logger(__name__)


class ModelCheckpoint:
    def __init__(self, save_dir='/root/checkpoint', monitor='val_loss', mode='min',
                 save_interval=3, save_best_only=True, verbose=True):
        """
        A callback to save model checkpoints during training.

        Args:
            save_dir (str): Directory to save model weights.
            monitor (str): The metric to monitor (e.g., 'val_loss', 'val_accuracy').
            mode (str): One of ['min', 'max']. Defines whether lower or higher metric is better.
            save_interval (int): Save checkpoint every N epochs.
            save_best_only (bool): If True, only save when the monitored metric improves.
            verbose (bool): If True, log checkpointing info.
        """
        assert mode in ['min', 'max'], "mode must be 'min' or 'max'"

        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_interval = save_interval
        self.save_best_only = save_best_only
        self.verbose = verbose

        os.makedirs(self.save_dir, exist_ok=True)

        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.cmp_op = (lambda current, best: current < best) if mode == 'min' else (lambda current, best: current > best)

        logger.info(f"Initialized ModelCheckpoint: monitor={monitor}, mode={mode}, "
                    f"interval={save_interval}, save_best_only={save_best_only}")

    def step(self, model, epoch, metrics: dict):
        """
        Call this method at the end of each epoch.

        Args:
            model (torch.nn.Module): The model being trained.
            epoch (int): Current epoch index (zero-based).
            metrics (dict): A dictionary containing monitored values, e.g., {'val_loss': 0.123}.
        """
        current_value = metrics.get(self.monitor)
        if current_value is None:
            if self.verbose:
                logger.warning(f"[ModelCheckpoint] Monitor '{self.monitor}' not found in metrics.")
            return

        save_this_epoch = (epoch + 1) % self.save_interval == 0
        improved = self.cmp_op(current_value, self.best_value)

        if self.save_best_only:
            if improved:
                self.best_value = current_value
                filename = os.path.join(self.save_dir, "best.pth")
                torch.save(model.state_dict(), filename)
                if self.verbose:
                    logger.info(f"[ModelCheckpoint] Saved improved model at epoch {epoch+1}: "
                                f"{self.monitor} = {current_value:.4f}")
        elif save_this_epoch:
            filename = os.path.join(self.save_dir, f"epoch{epoch+1}.pth")
            torch.save(model.state_dict(), filename)
            if self.verbose:
                logger.info(f"[ModelCheckpoint] Saved checkpoint at epoch {epoch+1}: "
                            f"{self.monitor} = {current_value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ModelCheckpoint callback")
    parser.add_argument('--epoch', type=int, default=4, help='Current epoch index')
    parser.add_argument('--val_loss', type=float, default=0.345, help='Current validation loss')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_test', help='Directory to save model')
    args = parser.parse_args()

    logger.info("Running CLI test for ModelCheckpoint")

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)

    dummy_model = DummyModel()
    metrics = {'val_loss': args.val_loss}

    callback = ModelCheckpoint(save_dir=args.save_dir, monitor='val_loss', mode='min',
                               save_interval=2, save_best_only=True, verbose=True)

    callback.step(dummy_model, args.epoch, metrics)
    logger.info("Test completed.")