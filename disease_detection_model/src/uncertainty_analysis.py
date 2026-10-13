import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model import RoboCropSoybeanV3, CLASS_NAMES
from predict import MODEL_PATH, DEVICE
from preprocessing import get_inference_transform


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_DIR = (
    r"C:\Users\Lenovo\Downloads\An India soyabean leaf dataset"
    r"\An India soyabean leaf dataset\Soyabean leaf"
    r"\soybean_disease_dataset_v3\validation"
)

BATCH_SIZE = 16


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("Loading model...")

    model = RoboCropSoybeanV3(num_classes=len(CLASS_NAMES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print("Device:", DEVICE)

    return model


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

def load_validation_dataset():

    if not os.path.exists(VALIDATION_DIR):
        raise FileNotFoundError(
            f"\nValidation directory not found:\n{VALIDATION_DIR}"
        )

    transform = get_inference_transform()

    dataset = datasets.ImageFolder(
        VALIDATION_DIR,
        transform=transform
    )

    print("\nValidation dataset:")
    print("Images:", len(dataset))
    print("Classes:", dataset.classes)
    print("Class mapping:", dataset.class_to_idx)

    return dataset


# ============================================================
# COLLECT PREDICTIONS
# ============================================================

@torch.no_grad()
def collect_predictions(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_labels = []
    all_predictions = []
    all_confidences = []
    all_margins = []

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)

        sorted_probabilities, sorted_indices = torch.sort(
            probabilities,
            descending=True,
            dim=1
        )

        top1_probability = sorted_probabilities[:, 0]
        top2_probability = sorted_probabilities[:, 1]

        top1_prediction = sorted_indices[:, 0]

        margin = top1_probability - top2_probability

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(top1_prediction.cpu().numpy())
        all_confidences.extend(top1_probability.cpu().numpy())
        all_margins.extend(margin.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_confidences),
        np.array(all_margins)
    )


# ============================================================
# BASIC VALIDATION RESULTS
# ============================================================

def print_basic_results(
    labels,
    predictions,
    confidences,
    margins
):

    accuracy = np.mean(predictions == labels)

    print("\n" + "=" * 70)
    print("VALIDATION UNCERTAINTY ANALYSIS")
    print("=" * 70)

    print(f"Validation images : {len(labels)}")
    print(f"Accuracy           : {accuracy * 100:.2f}%")

    print("\nConfidence statistics:")
    print(f"Mean   : {np.mean(confidences) * 100:.2f}%")
    print(f"Median : {np.median(confidences) * 100:.2f}%")
    print(f"Min    : {np.min(confidences) * 100:.2f}%")
    print(f"Max    : {np.max(confidences) * 100:.2f}%")

    print("\nTop-1 vs Top-2 margin statistics:")
    print(f"Mean   : {np.mean(margins) * 100:.2f}%")
    print(f"Median : {np.median(margins) * 100:.2f}%")
    print(f"Min    : {np.min(margins) * 100:.2f}%")
    print(f"Max    : {np.max(margins) * 100:.2f}%")


# ============================================================
# CONFIDENCE THRESHOLD ANALYSIS
# ============================================================

def analyze_confidence_thresholds(
    labels,
    predictions,
    confidences
):

    print("\n" + "=" * 70)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 70)

    thresholds = [
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]

    print(
        f"{'Threshold':<12}"
        f"{'Accepted':<12}"
        f"{'Rejected':<12}"
        f"{'Coverage':<12}"
        f"{'Accuracy':<12}"
    )

    print("-" * 60)

    for threshold in thresholds:

        accepted = confidences >= threshold

        accepted_count = np.sum(accepted)
        rejected_count = len(labels) - accepted_count

        if accepted_count > 0:
            accepted_accuracy = np.mean(
                predictions[accepted] == labels[accepted]
            )
        else:
            accepted_accuracy = 0.0

        coverage = accepted_count / len(labels)

        print(
            f"{threshold * 100:>7.0f}%     "
            f"{accepted_count:<12}"
            f"{rejected_count:<12}"
            f"{coverage * 100:>6.2f}%     "
            f"{accepted_accuracy * 100:>7.2f}%"
        )


# ============================================================
# MARGIN ANALYSIS
# ============================================================

def analyze_margin_thresholds(
    labels,
    predictions,
    margins
):

    print("\n" + "=" * 70)
    print("TOP-1 vs TOP-2 MARGIN ANALYSIS")
    print("=" * 70)

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30
    ]

    print(
        f"{'Margin':<12}"
        f"{'Accepted':<12}"
        f"{'Rejected':<12}"
        f"{'Coverage':<12}"
        f"{'Accuracy':<12}"
    )

    print("-" * 60)

    for threshold in thresholds:

        accepted = margins >= threshold

        accepted_count = np.sum(accepted)
        rejected_count = len(labels) - accepted_count

        if accepted_count > 0:
            accepted_accuracy = np.mean(
                predictions[accepted] == labels[accepted]
            )
        else:
            accepted_accuracy = 0.0

        coverage = accepted_count / len(labels)

        print(
            f"{threshold * 100:>7.0f}%     "
            f"{accepted_count:<12}"
            f"{rejected_count:<12}"
            f"{coverage * 100:>6.2f}%     "
            f"{accepted_accuracy * 100:>7.2f}%"
        )


# ============================================================
# COMBINED DECISION ANALYSIS
# ============================================================

def analyze_combined_rule(
    labels,
    predictions,
    confidences,
    margins
):

    print("\n" + "=" * 70)
    print("COMBINED CONFIDENCE + MARGIN ANALYSIS")
    print("=" * 70)

    rules = [
        (0.50, 0.05),
        (0.50, 0.10),
        (0.60, 0.10),
        (0.60, 0.15),
        (0.70, 0.10),
        (0.70, 0.15),
        (0.70, 0.20),
        (0.80, 0.15),
        (0.80, 0.20),
    ]

    print(
        f"{'Confidence':<14}"
        f"{'Margin':<12}"
        f"{'Accepted':<12}"
        f"{'Rejected':<12}"
        f"{'Coverage':<12}"
        f"{'Accuracy':<12}"
    )

    print("-" * 72)

    for confidence_threshold, margin_threshold in rules:

        accepted = (
            (confidences >= confidence_threshold)
            &
            (margins >= margin_threshold)
        )

        accepted_count = np.sum(accepted)
        rejected_count = len(labels) - accepted_count

        if accepted_count > 0:
            accepted_accuracy = np.mean(
                predictions[accepted] == labels[accepted]
            )
        else:
            accepted_accuracy = 0.0

        coverage = accepted_count / len(labels)

        print(
            f"{confidence_threshold * 100:>7.0f}%       "
            f"{margin_threshold * 100:>5.0f}%     "
            f"{accepted_count:<12}"
            f"{rejected_count:<12}"
            f"{coverage * 100:>6.2f}%     "
            f"{accepted_accuracy * 100:>7.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ROBOCROP SOYBEAN DISEASE - UNCERTAINTY ANALYSIS")
    print("=" * 70)

    model = load_model()

    dataset = load_validation_dataset()

    labels, predictions, confidences, margins = (
        collect_predictions(model, dataset)
    )

    print_basic_results(
        labels,
        predictions,
        confidences,
        margins
    )

    analyze_confidence_thresholds(
        labels,
        predictions,
        confidences
    )

    analyze_margin_thresholds(
        labels,
        predictions,
        margins
    )

    analyze_combined_rule(
        labels,
        predictions,
        confidences,
        margins
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(
        "\nIMPORTANT:"
        "\nThese results are for selecting a safe uncertainty rule."
        "\nDo NOT modify the model or test-set results based on this yet."
    )


if __name__ == "__main__":
    main()