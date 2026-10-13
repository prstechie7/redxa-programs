#!/usr/bin/env python3
"""
Production inference module for YOLO11n soybean pest detection model.

Classes:
0: Eocanthecona_Bug
1: Larva_Spodoptera
2: Red_Hairy_Catterpillar
3: Tobacco_Caterpillar
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# Default model path (portable relative path or from environment)
DEFAULT_MODEL_PATH = Path(
    os.getenv("YOLO_MODEL_PATH", "models/best.pt")
)

# Class names matching the training dataset
CLASS_NAMES = [
    "Eocanthecona_Bug",
    "Larva_Spodoptera",
    "Red_Hairy_Catterpillar",
    "Tobacco_Caterpillar"
]

# Output directory for annotated images
PREDICTIONS_DIR = Path(
    os.getenv("PREDICTIONS_DIR", "results/predictions")
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_device(device_arg: Optional[str] = None) -> str:
    """
    Determine the device to use for inference.
    
    Args:
        device_arg: Optional device string from CLI (e.g., "0", "cpu")
    
    Returns:
        Device string for Ultralytics ("0", "cpu", etc.)
    """
    if device_arg:
        return device_arg
    
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def validate_model_path(model_path: Path) -> Path:
    """Validate that the model file exists."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Expected production model at: {DEFAULT_MODEL_PATH}\n"
            f"Please train the model first or provide a valid model path."
        )
    return model_path


def validate_image_path(image_path: Path) -> Path:
    """Validate that the input image exists and is readable."""
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    if not image_path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if image_path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}\n"
            f"Supported formats: {', '.join(valid_extensions)}"
        )
    
    return image_path


def ensure_output_dir(output_dir: Path) -> Path:
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================
# CORE PREDICTION FUNCTION
# ============================================================

def predict_image(
    image_path: str,
    model_path: Optional[str] = None,
    confidence: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run pest detection inference on a single image.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained YOLO model (default: DEFAULT_MODEL_PATH)
        confidence: Confidence threshold for detections (default: 0.25)
        imgsz: Inference image size (default: 640)
        device: Device to run inference on (default: auto-detect CUDA)
    
    Returns:
        Dictionary with detection results:
        {
            "pest_detected": bool,
            "pest_count": int,
            "pests": [
                {
                    "class": str,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2]
                }
            ]
        }
    
    Raises:
        FileNotFoundError: If model or image not found
        ValueError: If image format unsupported
        RuntimeError: If inference fails
    """
    # Resolve and validate paths
    model_file = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    image_file = Path(image_path)
    
    validate_model_path(model_file)
    validate_image_path(image_file)
    
    # Determine device
    device_str = get_device(device)
    
    # Load model
    try:
        model = YOLO(str(model_file))
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Run inference
    try:
        results = model.predict(
            source=str(image_file),
            conf=confidence,
            imgsz=imgsz,
            device=device_str,
            verbose=False,
            save=False
        )
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")
    
    # Process results
    pests = []
    
    for result in results:
        if result.boxes is None:
            continue
        
        boxes = result.boxes
        for i in range(len(boxes)):
            # Get class ID and name
            cls_id = int(boxes.cls[i].item())
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            
            # Get confidence
            conf = float(boxes.conf[i].item())
            
            # Get bounding box [x1, y1, x2, y2]
            xyxy = boxes.xyxy[i].tolist()
            bbox = [float(coord) for coord in xyxy]
            
            pests.append({
                "class": class_name,
                "confidence": conf,
                "bbox": bbox
            })
    
    # Build response
    pest_count = len(pests)
    
    return {
        "pest_detected": pest_count > 0,
        "pest_count": pest_count,
        "pests": pests
    }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Soybean Pest Detection Inference - YOLO11n",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained YOLO model"
    )
    
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (e.g., '0', 'cpu'). Auto-detects if not specified."
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated image with bounding boxes"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PREDICTIONS_DIR),
        help="Directory to save annotated images"
    )
    
    return parser.parse_args()


def save_annotated_image(
    image_path: Path,
    output_dir: Path,
    result: Dict[str, Any]
) -> Path:
    """
    Save annotated image using Ultralytics built-in plotting.
    
    Note: This re-runs prediction with save=True to get the annotated image.
    """
    # For simplicity, we use the model's save functionality
    # We need to run prediction again with save=True
    model = YOLO(str(DEFAULT_MODEL_PATH))
    
    output_dir = ensure_output_dir(output_dir)
    
    # Generate output filename
    output_name = f"pred_{image_path.stem}{image_path.suffix}"
    output_path = output_dir / output_name
    
    # Run with save=True to generate annotated image
    _ = model.predict(
        source=str(image_path),
        conf=0.25,  # Will be overridden by actual threshold in practice
        imgsz=640,
        device=get_device(),
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        verbose=False
    )
    
    # Find the saved file (Ultralytics creates a subdirectory)
    saved_files = list(output_dir.glob(f"*{image_path.stem}*"))
    if saved_files:
        return saved_files[0]
    
    return output_path


def print_results(result: Dict[str, Any]):
    """Print detection results to console."""
    print("\n" + "=" * 50)
    print("PEST DETECTION RESULTS")
    print("=" * 50)
    
    if result["pest_detected"]:
        print(f"Pest DETECTED: Yes")
        print(f"Total pests found: {result['pest_count']}")
        print()
        
        for i, pest in enumerate(result["pests"], 1):
            print(f"  Detection #{i}:")
            print(f"    Class:      {pest['class']}")
            print(f"    Confidence: {pest['confidence']:.4f}")
            print(f"    BBox:       [{pest['bbox'][0]:.1f}, {pest['bbox'][1]:.1f}, "
                  f"{pest['bbox'][2]:.1f}, {pest['bbox'][3]:.1f}]")
            print()
    else:
        print("Pest DETECTED: No")
        print("No pests found in the image.")
    
    print("=" * 50)


def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    try:
        # Run prediction
        result = predict_image(
            image_path=args.image,
            model_path=args.model,
            confidence=args.confidence,
            imgsz=args.imgsz,
            device=args.device
        )
        
        # Print results
        print_results(result)
        
        # Save annotated image if requested
        if args.save:
            print(f"\nSaving annotated image to: {args.output_dir}")
            try:
                image_path = Path(args.image)
                output_dir = Path(args.output_dir)
                ensure_output_dir(output_dir)
                
                # Use Ultralytics save functionality
                model = YOLO(str(args.model) if args.model else str(DEFAULT_MODEL_PATH))
                
                saved_results = model.predict(
                    source=str(image_path),
                    conf=args.confidence,
                    imgsz=args.imgsz,
                    device=get_device(args.device),
                    save=True,
                    project=str(output_dir.parent),
                    name=output_dir.name,
                    exist_ok=True,
                    verbose=False
                )
                
                # Find saved file
                for r in saved_results:
                    if hasattr(r, 'save_dir'):
                        saved_dir = Path(r.save_dir)
                        saved_files = list(saved_dir.glob("*"))
                        if saved_files:
                            print(f"Annotated image saved to: {saved_files[0]}")
                            break
                        
            except Exception as e:
                print(f"Warning: Could not save annotated image: {e}")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()