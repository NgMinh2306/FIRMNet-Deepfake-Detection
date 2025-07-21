# data/transforms.py

import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import argparse
import json
import yaml
from types import SimpleNamespace

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger

logger = get_logger(__name__)

def divide_255(image, **kwargs):
    return (image / 255.0).astype('float32')

def get_train_transforms(opt):
    return A.Compose([
        A.SmallestMaxSize(max_size=opt.resize_max),
        A.RandomCrop(height=opt.crop_height, width=opt.crop_width),
        A.HorizontalFlip(p=opt.hflip_prob),

        A.Affine(
            translate_percent=opt.affine_trans,
            scale=(1 - opt.affine_scale, 1 + opt.affine_scale),
            rotate=(-opt.affine_rotate, opt.affine_rotate),
            p=opt.affine_p
        ),

        A.OneOf([
            A.MotionBlur(blur_limit=opt.blur_limit),
            A.MedianBlur(blur_limit=opt.blur_limit),
            A.GaussianBlur(blur_limit=opt.blur_limit)
        ], p=opt.blur_p),

        A.OneOf([
            A.GaussNoise(std_range=(opt.noise_std_min, opt.noise_std_max)),
            A.ISONoise()
        ], p=opt.noise_p),

        A.RandomBrightnessContrast(p=opt.brightness_contrast_p),
        A.HueSaturationValue(p=opt.hsv_p),
        A.RGBShift(p=opt.rgbshift_p),
        A.CLAHE(p=opt.clahe_p),

        A.Downscale(
            scale_range=tuple(opt.downscale_range),
            p=opt.downscale_p
        ),

        A.ImageCompression(
            quality_range=tuple(opt.jpeg_quality),
            compression_type='jpeg',
            p=opt.jpeg_p
        ),

        A.CoarseDropout(
            num_holes_range=tuple(opt.coarse_num_holes),
            hole_height_range=tuple(opt.coarse_h_range),
            hole_width_range=tuple(opt.coarse_w_range),
            p=opt.coarse_p
        ),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if opt.use_imagenet_norm
        else A.Lambda(image=divide_255, name="normalize_0_1"),

        ToTensorV2()
    ])


def get_valid_transforms():
    return A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=224, width=224),
        A.Lambda(image=divide_255, name="normalize_0_1_valid"),
        ToTensorV2()
    ])


class AlbumentationsTransform:
    def __init__(self, albumentations_transform):
        self.albumentations_transform = albumentations_transform

    def __call__(self, img):
        image = np.array(img)
        return self.albumentations_transform(image=image)["image"]


def apply_shortcut_flags(opt):
    if getattr(opt, 'augment_strong', False):
        opt.blur_p = 0.3
        opt.noise_p = 0.2
        opt.downscale_p = 0.2
        opt.jpeg_p = 0.2
        opt.coarse_p = 0.3
        logger.info("Applied --augment_strong settings")

    if getattr(opt, 'color_jitter', False):
        opt.brightness_contrast_p = 0.3
        opt.hsv_p = 0.2
        opt.rgbshift_p = 0.2
        opt.clahe_p = 0.2
        logger.info("Applied --color_jitter settings")


def load_config(config_path):
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported config format: only .json or .yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None, help="Path to config .json or .yaml file")

    # Shortcut flags
    parser.add_argument('--augment_strong', action='store_true', help="Apply strong augmentation defaults")
    parser.add_argument('--color_jitter', action='store_true', help="Apply color jitter defaults")

    # Transform arguments
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

    parser.add_argument('--noise_std_min', type=float, default=10 / 255)
    parser.add_argument('--noise_std_max', type=float, default=50 / 255)
    parser.add_argument('--noise_p', type=float, default=0.0)

    parser.add_argument('--brightness_contrast_p', type=float, default=0.0)
    parser.add_argument('--hsv_p', type=float, default=0.0)
    parser.add_argument('--rgbshift_p', type=float, default=0.0)
    parser.add_argument('--clahe_p', type=float, default=0.0)

    parser.add_argument('--downscale_range', nargs=2, type=float, default=[0.25, 0.75])
    parser.add_argument('--downscale_p', type=float, default=0.0)

    parser.add_argument('--jpeg_quality', nargs=2, type=int, default=[30, 80])
    parser.add_argument('--jpeg_p', type=float, default=0.0)

    parser.add_argument('--coarse_num_holes', nargs=2, type=int, default=[3, 6])
    parser.add_argument('--coarse_h_range', nargs=2, type=float, default=[0.05, 0.15])
    parser.add_argument('--coarse_w_range', nargs=2, type=float, default=[0.05, 0.15])
    parser.add_argument('--coarse_p', type=float, default=0.0)

    parser.add_argument('--use_imagenet_norm', action='store_true')

    args = parser.parse_args()

    # Load from config file if provided
    if args.config:
        config_dict = load_config(args.config)
        for k, v in config_dict.items():
            setattr(args, k, v)
        logger.info(f"Loaded config from: {args.config}")

    # Apply shortcut logic
    apply_shortcut_flags(args)

    logger.info("Final transform config:")
    logger.info(json.dumps(vars(args), indent=2))

    train_transform = get_train_transforms(args)
    valid_transform = get_valid_transforms()

    logger.info("Albumentations transforms created successfully.")