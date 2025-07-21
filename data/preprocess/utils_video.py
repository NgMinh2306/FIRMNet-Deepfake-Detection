# data/preprocess/utils_video.py

import os
import argparse
import cv2
import numpy as np
from deepface import DeepFace

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from logger import get_logger

logger = get_logger(__name__)

def select_frames_with_gap(total, num_frames, min_gap):
    """
    Randomly select frames ensuring minimum spacing between frames.
    """
    attempts = 0
    while attempts < 10:
        indices = sorted(np.random.choice(range(total), size=num_frames, replace=False))
        if all(indices[i+1] - indices[i] >= min_gap for i in range(len(indices)-1)):
            return indices
        attempts += 1
    return np.linspace(0, total - 1, num=num_frames, dtype=int)

def process_video(
    video_path,
    output_folder,
    video_name,
    resize_dim=(224, 224),
    margin_percent=40.0,
    num_frames=10,
    min_frame_gap=10
):
    """
    Extract faces from video and save resized images to output folder.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder to save output images.
        video_name (str): Video name without extension (used as image prefix).
        resize_dim (tuple): Resize dimensions (W, H).
        margin_percent (float): Margin percentage around the face.
        num_frames (int): Number of frames to extract.
        min_frame_gap (int): Minimum gap between frames.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        logger.warning(f"Skipped {video_name}: total_frames = 0")
        return

    if total_frames < num_frames:
        frame_indices = np.sort(np.random.choice(range(total_frames), size=num_frames, replace=True))
    else:
        frame_indices = select_frames_with_gap(total_frames, num_frames, min_frame_gap)

    logger.debug(f"{video_name}: selected frame indices = {frame_indices.tolist()}")

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"{video_name} - frame {i}: could not read frame")
            continue

        try:
            results = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='yolov8',
                enforce_detection=False,
                align=False,
                expand_percentage=margin_percent
            )
            if results:
                face = results[0]
                facial_area = face.get("facial_area", None)
                if facial_area:
                    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, resize_dim)
                    frame_name = f"{video_name}_frame{i}.jpg"
                    save_path = os.path.join(output_folder, frame_name)
                    cv2.imwrite(save_path, face_img)
                    logger.debug(f"Saved: {save_path}")
            else:
                logger.warning(f"{video_name} - frame {i}: no face detected")

        except Exception as e:
            logger.error(f"{video_name} - frame {i}: {e}")

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test processing one video.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to a .mp4 video")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for extracted faces")
    parser.add_argument("--resize_dim", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--margin_percent", type=float, default=40.0)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--min_frame_gap", type=int, default=10)

    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]

    os.makedirs(args.output_folder, exist_ok=True)

    process_video(
        video_path=args.video_path,
        output_folder=args.output_folder,
        video_name=video_name,
        resize_dim=tuple(args.resize_dim),
        margin_percent=args.margin_percent,
        num_frames=args.num_frames,
        min_frame_gap=args.min_frame_gap
    )