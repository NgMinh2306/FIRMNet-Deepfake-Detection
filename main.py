# =============================
# 1. Load Dataset & Augment
# =============================
from logger import get_logger
from data.dataloader import get_dataloaders, load_config
from data.transforms import apply_shortcut_flags
import argparse
import json
from types import SimpleNamespace

# ==========================
# 2. Build Model (CNN + FAGate + FER)
# ==========================
from models.backbones import EfficientNetV2S
import torch

# =============================
# 3. Loss, Optimizer, Scheduler, Callback, Train Loop
# =============================
from train.losses import FocalLoss
from train.optimizer import choose_optimizer, get_scheduler
from train.callbacks import ModelCheckpoint
from train.trainer import train_model

logger = get_logger(__name__, log_dir="logs", log_filename="main.log")

def load_data(cfg_path):
    # Load config (yaml or json)
    logger.info(f"Loading data config from {cfg_path}")
    config_dict = load_config(cfg_path)
    args = argparse.Namespace(**config_dict)

    # Apply augmentation shortcut flags
    apply_shortcut_flags(args)

    # Log final augmentation config
    logger.info("Final data config:")
    logger.info(json.dumps(vars(args), indent=2))

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        transform_opt=args,
        logger=logger
    )

    return train_loader, val_loader, args

def build_model(device: torch.device, num_classes=2, pretrained_path=None):
    logger.info("Initializing model: EfficientNetV2-S + FAGate + FER")

    model = EfficientNetV2S(num_classes=num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if pretrained_path:
        logger.info(f"Loading pretrained weights from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Pretrained weights loaded.")

    return model

def setup_training(model, train_loader, val_loader, device, num_epochs=20, use_amp=True):
    # Loss function
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')

    # Optimizer
    optimizer = choose_optimizer(
        model,
        optimizer_type='adam',
        lr=1e-4,
        weight_decay=1e-6
    )

    # Scheduler
    scheduler_config = {
        'scheduler_type': 'cosine',
        'warmup_type': 'linear',
        'warmup_epochs': 3,
        'scheduler_params': {}
    }

    # ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        save_dir='checkpoints',
        monitor='val_loss',
        mode='min',
        save_interval=3,
        save_best_only=False,
        verbose=True
    )

    # Train loop
    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        metric_add=['AP', 'F1 score', 'precision', 'recall', 'EER'],
        checkpoint_callback=checkpoint_callback,
        use_amp=use_amp,
        use_scheduler=True,
        scheduler_config=scheduler_config,
        logger=logger
    )
    
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FAGNet Training Pipeline")

    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to .json or .yaml config file")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained weights (optional)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (override config)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="Device to train on")

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

    parser.add_argument('--opt', type=str, default='adam', help="Optimizer type: adam | sgd | adamw")
    parser.add_argument('--sched', type=str, default='cosine', help="Scheduler type")
    parser.add_argument('--warmup', type=str, default='linear', help="Warmup type: linear | cosine")
    parser.add_argument('--warmup_epochs', type=int, default=3, help="Number of warmup epochs")
    parser.add_argument('--soft', action='store_true', help='Use SoftFocalLoss instead of FocalLoss')
    parser.add_argument('--metrics', type=str, nargs='*', default=['AP', 'F1 score', 'precision', 'recall', 'EER'], help="Extra metrics to compute")

    args = parser.parse_args()

    # Load from config file if provided
    if args.config and args.config.endswith((".yaml", ".yml", ".json")):
        logger.info(f"Loading config from: {args.config}")
        cfg = load_config(args.config)
        for k, v in cfg.items():
            setattr(args, k, v)
        logger.info("Final configuration (from config file):")
    else:
        logger.info("Using manually provided command-line arguments:")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, data_args = load_data(args.config)
    
    # Optional override
    pretrained_path = args.pretrained or getattr(data_args, "pretrained_path", None)
    num_epochs = args.epochs or getattr(data_args, "num_epochs", 20)
    
    # Build model and train
    model = build_model(device, num_classes=2, pretrained_path=pretrained_path)
    history = setup_training(model, train_loader, val_loader, device, num_epochs=num_epochs)
    
    # Log training history
    logger.info("Training history from main: \n%s", history)
