# data/preprocess/extract_frames_dfdc.py

import os
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from logger import get_logger
from utils_video import process_video

logger = get_logger(__name__)

def extract_faces_from_dfdc(
    metadata_path,
    video_root_dir,
    image_root_dir,
    num_frames=10,
    min_frame_gap=10,
    margin_percent=40.0,
    resize_dim=(224, 224),
    fake_real_ratio=None
):
    os.makedirs(image_root_dir, exist_ok=True)
    for label in ["FAKE", "REAL"]:
        os.makedirs(os.path.join(image_root_dir, label), exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    video_files = []
    for subdir in os.listdir(video_root_dir):
        full_path = os.path.join(video_root_dir, subdir)
        if os.path.isdir(full_path) and subdir.startswith("dfdc_train_part"):
            video_files.extend([
                file for file in os.listdir(full_path) if file.endswith(".mp4")
            ])

    filtered_metadata = metadata[metadata['filename'].isin(video_files)].copy()
    filtered_metadata['label'] = filtered_metadata['label'].str.strip().str.upper()

    real_videos = filtered_metadata[filtered_metadata['label'] == 'REAL']
    fake_videos = filtered_metadata[filtered_metadata['label'] == 'FAKE']

    if fake_real_ratio is not None:
        real_count = len(real_videos)
        fake_target_count = int(real_count * fake_real_ratio)

        if fake_target_count > len(fake_videos):
            logger.warning(f"Not enough FAKE videos for target ratio {fake_real_ratio}. Using {len(fake_videos)} instead.")
            fake_target_count = len(fake_videos)

        fake_videos = fake_videos.sample(n=fake_target_count, random_state=42)
        df = pd.concat([real_videos, fake_videos], ignore_index=True)
    else:
        df = filtered_metadata

    logger.info(f"Using {len(real_videos)} REAL and {len(fake_videos)} FAKE videos (ratio={fake_real_ratio}).")

    label_map = dict(zip(df['filename'], df['label']))
    video_folders = glob(os.path.join(video_root_dir, "dfdc_train_part_*"))

    for folder in video_folders:
        logger.info(f"Processing folder: {os.path.basename(folder)}")
        for video_path in tqdm(glob(os.path.join(folder, "*.mp4")), desc=f"[{os.path.basename(folder)}]", unit="video"):
            video_file = os.path.basename(video_path)
            if video_file not in label_map:
                continue

            label = label_map[video_file]
            label_folder = os.path.join(image_root_dir, label)
            video_name = os.path.splitext(video_file)[0]

            process_video(
                video_path=video_path,
                output_folder=label_folder,
                video_name=video_name,
                resize_dim=resize_dim,
                margin_percent=margin_percent,
                num_frames=num_frames,
                min_frame_gap=min_frame_gap
            )

    logger.info("Extraction completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from DFDC videos using DeepFace.")
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--video_root_dir", type=str, required=True)
    parser.add_argument("--image_root_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--min_frame_gap", type=int, default=10)
    parser.add_argument("--margin_percent", type=float, default=40.0)
    parser.add_argument("--resize_dim", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--fake_real_ratio", type=float, default=None)

    args = parser.parse_args()

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