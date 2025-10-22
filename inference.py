# inference.py

import torch
import torch.nn.functional as F
import cv2
import os
from PIL import Image
from deepface import DeepFace
from logger import get_logger
from datetime import datetime
from models.backbones import FIRMNet
from torchvision import transforms

def run_inference(pretrained_path, image_path, device, margin_percent, num_classes=2, detector_backend='yolov8'):
    """
    Perform inference on a single image to classify it as FAKE or REAL using FIRMNet.

    Args:
        pretrained_path (str): Path to the pretrained model weights.
        image_path (str): Path to the input image for inference.
        device (str): Device to run inference on ('cuda' or 'cpu').
        margin_percent (float): Percentage to expand the face detection area.
        num_classes (int, optional): Number of output classes (default: 2 for FAKE/REAL).
        detector_backend (str, optional): Face detection backend (default: 'yolov8').
    
    Raises:
        RuntimeError: If image loading, face detection, or facial area extraction fails.
    """
    # Initialize logger
    logger = get_logger(__name__, log_dir="logs", log_filename=f"inference_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    
    # Load model
    logger.info("Loading model...")
    model = FIRMNet(num_classes=num_classes).to(device)
    with torch.no_grad():
        _ = model(torch.randn(1, 3, 224, 224).to(device))

    # Load pre-trained weights
    logger.info(f"Loading weights from: {pretrained_path}")
    state = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")

    # Load and crop face from image
    logger.info(f"Reading image from: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Failed to read image: {image_path}")
        raise RuntimeError(f"Failed to read image: {image_path}")

    # Detect face
    logger.info("Detecting face...")
    results_face = DeepFace.extract_faces(
        img_path=frame,
        detector_backend=detector_backend,
        enforce_detection=False,
        align=False,
        expand_percentage=margin_percent
    )

    if not results_face:
        logger.error("No face detected!")
        raise RuntimeError("No face detected!")

    facial_area = results_face[0].get("facial_area", None)
    if not facial_area:
        logger.error("Failed to retrieve facial_area!")
        raise RuntimeError("Failed to retrieve facial_area!")

    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
    logger.info(f"Face detected at: x={x}, y={y}, w={w}, h={h}")
    face_crop = frame[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop, (224, 224))
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)
    logger.info("Face cropped and resized successfully.")

    # Transform and perform inference
    logger.info("Applying transform and performing inference...")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]  # [FAKE, REAL]
        predicted_class = probs.argmax()
        confidence = probs[predicted_class]

    label_map = {0: "FAKE", 1: "REAL"}
    logger.info(f"Prediction: {label_map[predicted_class]} (confidence: {confidence:.4f})")
    logger.info(f"Softmax output: FAKE = {probs[0]:.4f}, REAL = {probs[1]:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform inference on an image using FIRMNet")
    parser.add_argument("--pretrained", type=str, default="checkpoints/epoch24.pth",
                        help="Path to pretrained weights (optional)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--margin_percent", type=float, default=40.0,
                        help="Percentage to expand face detection area")
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = get_logger(__name__, log_dir="logs", log_filename=f"inference_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
    logger.info(f"Starting inference with pretrained={args.pretrained}, device={args.device}, image={args.image_path}, margin={args.margin_percent}")
    
    # Run inference
    run_inference(
        pretrained_path=args.pretrained,
        image_path=args.image_path,
        device=args.device,
        margin_percent=args.margin_percent
    )