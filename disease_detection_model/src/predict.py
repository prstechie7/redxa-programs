import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model import RoboCropSoybeanV3, CLASS_NAMES
from preprocessing import get_inference_transform


# ============================================================
# PATHS AND DEVICE
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(CURRENT_DIR)
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "disease_detection_model",
    "models",
    "robocrop_soybean_v3_final.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# UNCERTAINTY DECISION THRESHOLDS
# ============================================================
#
# These thresholds were selected from the validation-set
# uncertainty analysis.
#
# IMPORTANT:
# They are NOT guarantees of real-world accuracy.
#

HIGH_MARGIN = 0.07
REVIEW_MARGIN = 0.05


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = RoboCropSoybeanV3(
        num_classes=len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# UNCERTAINTY DECISION
# ============================================================

def determine_decision(margin):

    if margin >= HIGH_MARGIN:

        return {
            "status": "CONFIDENT",
            "action": "Accept prediction",
            "message": (
                "Prediction has a strong Top-1 vs Top-2 margin."
            )
        }

    elif margin >= REVIEW_MARGIN:

        return {
            "status": "REVIEW",
            "action": "Capture another image",
            "message": (
                "Prediction is borderline. "
                "Capture another image before making a final decision."
            )
        }

    else:

        return {
            "status": "UNCERTAIN",
            "action": "Capture another image",
            "message": (
                "Model is uncertain between the leading classes. "
                "Do not use this prediction as a confirmed diagnosis."
            )
        }


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(model, image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    transform = get_inference_transform()

    image_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

    sorted_indices = torch.argsort(
        probabilities,
        descending=True
    )

    top_predictions = []

    for index in sorted_indices:

        index = index.item()

        probability = probabilities[index].item()

        top_predictions.append(
            {
                "disease": CLASS_NAMES[index],
                "confidence": probability,
                "confidence_percent": (
                    probability * 100
                )
            }
        )

    # --------------------------------------------------------
    # TOP-1
    # --------------------------------------------------------

    top1 = top_predictions[0]

    # --------------------------------------------------------
    # TOP-2
    # --------------------------------------------------------

    top2 = top_predictions[1]

    # --------------------------------------------------------
    # TOP-1 / TOP-2 MARGIN
    # --------------------------------------------------------

    margin = (
        top1["confidence"]
        -
        top2["confidence"]
    )

    margin_percent = (
        margin * 100
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision = determine_decision(
        margin
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {

        "disease": top1["disease"],

        "confidence": top1["confidence"],

        "confidence_percent":
            top1["confidence_percent"],

        "top2_disease":
            top2["disease"],

        "top2_confidence":
            top2["confidence"],

        "top2_confidence_percent":
            top2["confidence_percent"],

        "margin":
            margin,

        "margin_percent":
            margin_percent,

        "status":
            decision["status"],

        "action":
            decision["action"],

        "message":
            decision["message"],

        "top_predictions":
            top_predictions
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "ROBOCROP SOYBEAN DISEASE DETECTION"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print()

    print(
        "Prediction module loaded successfully."
    )

    print()
    print(
        "Uncertainty policy:"
    )

    print(
        f"  Margin >= {HIGH_MARGIN * 100:.0f}%"
        "  -> CONFIDENT"
    )

    print(
        f"  Margin >= {REVIEW_MARGIN * 100:.0f}%"
        f" and < {HIGH_MARGIN * 100:.0f}%"
        " -> REVIEW"
    )

    print(
        f"  Margin < {REVIEW_MARGIN * 100:.0f}%"
        " -> UNCERTAIN"
    )
