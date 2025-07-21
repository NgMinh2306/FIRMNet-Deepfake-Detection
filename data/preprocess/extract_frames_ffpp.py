# data/preprocess/extract_frames_ffpp.py

import os
import argparse
from glob import glob
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from logger import get_logger
from utils_video import process_video

logger = get_logger(__name__)

def extract_faces_ffpp(
    video_root_dir,
    image_root_dir,
    num_frames=10,
    min_frame_gap=10,
    margin_percent=40.0,
    resize_dim=(224, 224)
):
    for label in ["real", "fake"]:
        label_upper = label.upper()
        video_folder = os.path.join(video_root_dir, label)
        output_folder = os.path.join(image_root_dir, label_upper)
        os.makedirs(output_folder, exist_ok=True)

        video_files = glob(os.path.join(video_folder, "*.mp4"))
        logger.info(f"Found {len(video_files)} videos in '{label}' folder")

        for video_path in tqdm(video_files, desc=f"[{label_upper}]", unit="video"):
            video_file = os.path.basename(video_path)
            video_name = os.path.splitext(video_file)[0]

            process_video(
                video_path=video_path,
                output_folder=output_folder,
                video_name=video_name,
                resize_dim=resize_dim,
                margin_percent=margin_percent,
                num_frames=num_frames,
                min_frame_gap=min_frame_gap
            )

    logger.info("Extraction completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from FF++ videos using YOLOv8 (DeepFace)")
    parser.add_argument("--video_root_dir", type=str, required=True, help="Path to 'real/' and 'fake/' folders")
    parser.add_argument("--image_root_dir", type=str, required=True, help="Path to save output cropped face images")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to extract per video")
    parser.add_argument("--min_frame_gap", type=int, default=10, help="Minimum gap between frames")
    parser.add_argument("--margin_percent", type=float, default=40.0, help="Margin around face in percent")
    parser.add_argument("--resize_dim", type=int, nargs=2, default=(224, 224), help="Resize dimensions for face (width height)")

    args = parser.parse_args()
    
    extract_faces_ffpp(
        video_root_dir=args.video_root_dir,
        image_root_dir=args.image_root_dir,
        num_frames=args.num_frames,
        min_frame_gap=args.min_frame_gap,
        margin_percent=args.margin_percent,
        resize_dim=tuple(args.resize_dim)
    )
