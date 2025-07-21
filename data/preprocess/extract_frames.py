# data/preprocess/extract_frames.py

import argparse
import yaml
import json
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from logger import get_logger
from extract_frames_dfdc import extract_faces_from_dfdc
from extract_frames_ffpp import extract_faces_ffpp

logger = get_logger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        if config_path.endswith(".json"):
            return json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported config file format. Use .yaml or .json")

def main():
    parser = argparse.ArgumentParser(description="Extract face frames from video datasets (DFDC, FF++)")

    # Optional config path
    parser.add_argument('--config', type=str, default=None, help="Path to config .yaml or .json file")

    # Manual override options (same as keys in config)
    parser.add_argument('--dataset', type=str, help="Dataset name: dfdc or ffpp")
    parser.add_argument('--metadata_path', type=str)
    parser.add_argument('--video_root_dir', type=str)
    parser.add_argument('--image_root_dir', type=str)
    parser.add_argument('--num_frames', type=int, default=10)
    parser.add_argument('--min_frame_gap', type=int, default=10)
    parser.add_argument('--margin_percent', type=float, default=40.0)
    parser.add_argument('--resize_dim', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--fake_real_ratio', type=float, default=None)

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
    
    logger.info(json.dumps(vars(args), indent=2))

    # Call correct extractor
    if args.dataset.lower() == 'dfdc':
        logger.info("Starting DFDC frame extraction...")
        extract_faces_from_dfdc(
            metadata_path=args.metadata_path,
            video_root_dir=args.video_root_dir,
            image_root_dir=args.image_root_dir,
            num_frames=args.num_frames,
            min_frame_gap=args.min_frame_gap,
            margin_percent=args.margin_percent,
            resize_dim=tuple(args.resize_dim),
            fake_real_ratio=args.fake_real_ratio
        )
    elif args.dataset.lower() == 'ffpp':
        logger.info("Starting FF++ frame extraction...")
        extract_faces_ffpp(
            video_root_dir=args.video_root_dir,
            image_root_dir=args.image_root_dir,
            num_frames=args.num_frames,
            min_frame_gap=args.min_frame_gap,
            margin_percent=args.margin_percent,
            resize_dim=tuple(args.resize_dim)
        )
    else:
        logger.error(f"Unsupported dataset: {args.dataset}. Use 'dfdc' or 'ffpp'.")

if __name__ == "__main__":
    main()