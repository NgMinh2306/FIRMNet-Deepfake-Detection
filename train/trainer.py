# train/trainer.py

import time
import numpy as np
import torch
try:
    from torch.amp import autocast, GradScaler # PyTorch >= 2.0
except ImportError:
    from torch.cuda.amp import autocast, GradScaler # PyTorch < 2.0

from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger
from train.optimizer import choose_optimizer, get_scheduler
from train.callbacks import ModelCheckpoint
from train.losses import FocalLoss
from train.metrics import compute_metrics

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs,
                device='cuda', metric_add=None, checkpoint_callback=None,
                use_amp=True, use_scheduler=False, scheduler_config=None, logger=None):
    """
    Train a binary classification model and track metrics per epoch.

    Args:
        model: PyTorch model.
        train_loader, valid_loader: DataLoaders.
        criterion: loss function.
        optimizer: torch optimizer.
        num_epochs (int): training epochs.
        device (str): 'cuda' or 'cpu'.
        metric_add (list[str]): additional metrics to compute.
        checkpoint_callback (ModelCheckpoint): optional checkpointing.
        use_amp (bool): use mixed precision.
        use_scheduler (bool): use scheduler.
        scheduler_config (dict): scheduler and warmup settings.

    Returns:
        history (dict): training and validation history.
    """

    logger = logger or get_logger(__name__)
    
    if isinstance(metric_add, str):
        metric_add = [metric_add]
    elif metric_add is None:
        metric_add = []

    model = model.to(device)
    scaler = GradScaler(enabled=use_amp)

    warmup_scheduler = None
    scheduler = None

    if use_scheduler:
        scheduler, warmup_scheduler = get_scheduler(
            optimizer,
            num_epochs=num_epochs,
            train_loader_len=len(train_loader),
            **scheduler_config
        )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_auc': [], 'val_auc': []
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        train_loss, y_train_true, y_train_prob = 0.0, [], []

        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_amp):  
                outputs = model(images)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            y_train_prob.extend(probs)
            y_train_true.extend(labels.cpu().numpy())

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        y_train_true = np.array(y_train_true)
        y_train_prob = np.array(y_train_prob)
        train_loss /= len(train_loader.dataset)
        metrics_train = compute_metrics(y_train_true, y_train_prob, extra_metrics=metric_add)

        model.eval()
        val_loss, y_val_true, y_val_prob = 0.0, [], []
        val_bar = tqdm(valid_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                y_val_prob.extend(probs)
                y_val_true.extend(labels.cpu().numpy())
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        y_val_true = np.array(y_val_true)
        y_val_prob = np.array(y_val_prob)
        val_loss /= len(valid_loader.dataset)
        metrics_val = compute_metrics(y_val_true, y_val_prob, extra_metrics=metric_add)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(metrics_train['accuracy'])
        history['val_accuracy'].append(metrics_val['accuracy'])
        history['train_auc'].append(metrics_train['auc'])
        history['val_auc'].append(metrics_val['auc'])

        # Logging summary
        epoch_time = time.time() - epoch_start
        logger.info(f"{len(train_loader)} steps - {int(epoch_time)}s/epoch "
                    f"- loss: {train_loss:.4f} - acc: {metrics_train['accuracy']:.4f} - auc: {metrics_train['auc']:.4f} "
                    f"- val_loss: {val_loss:.4f} - val_acc: {metrics_val['accuracy']:.4f} - val_auc: {metrics_val['auc']:.4f}")
        for m in metric_add:
            if m in metrics_train:
                logger.info(f" - train_{m}: {metrics_train[m]:.4f} - val_{m}: {metrics_val[m]:.4f}")

        # LR scheduler step
        if use_scheduler:
            if warmup_scheduler and epoch < scheduler_config.get('warmup_epochs', 0):
                warmup_scheduler.step()
            elif scheduler:
                if scheduler_config.get('scheduler_type') == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

        # Checkpoint
        if checkpoint_callback:
            checkpoint_callback.step(model, epoch, {
                'val_loss': val_loss,
                'val_accuracy': metrics_val['accuracy']
            })

    return history


if __name__ == "__main__":
    print("Python path:", sys.executable)
    print("Torch version:", torch.__version__)
    
    import argparse

    parser = argparse.ArgumentParser(description="Run training CLI")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--sched', type=str, default='cosine')
    parser.add_argument('--warmup', type=str, default='linear')
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--soft', action='store_true', help='Use SoftFocalLoss instead of FocalLoss')
    parser.add_argument('--metrics', type=str, nargs='*', default=['AP', 'F1 score', 'precision', 'recall', 'EER'])
    args = parser.parse_args()

    logger = get_logger(__name__)
    logger.info("Running CLI test for trainer.py")

    # Dummy model and data
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 2)

        def forward(self, x):
            return self.linear(x)

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, n=100):
            self.x = torch.randn(n, 16)
            self.y = torch.randint(0, 2, (n,))

        def __len__(self): return len(self.x)
        def __getitem__(self, idx): return self.x[idx], self.y[idx]

    model = DummyModel()
    train_loader = torch.utils.data.DataLoader(DummyDataset(100), batch_size=args.batch)
    val_loader = torch.utils.data.DataLoader(DummyDataset(40), batch_size=args.batch)

    optimizer = choose_optimizer(model, optimizer_type='adam')
    criterion = FocalLoss(alpha=torch.tensor([1.0, 1.0]))
    checkpoint_callback = ModelCheckpoint(save_dir='./test_ckpt', monitor='val_loss')

    scheduler_config = {
        'scheduler_type': 'cosine',
        'warmup_type': 'linear',
        'warmup_epochs': 2,
        'scheduler_params': {}
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=args.epochs,
                device=device,
                metric_add=['AP', 'F1 score', 'precision', 'recall', 'EER'],
                checkpoint_callback=checkpoint_callback,
                use_amp=False,
                use_scheduler=True,
                scheduler_config=scheduler_config)
    
    logger.info("Training history: %s", history)
