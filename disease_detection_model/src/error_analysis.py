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

    model = RoboCropSoybeanV3(
        num_classes=len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print("Device:", DEVICE)

    return model


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation_dataset():

    if not os.path.exists(VALIDATION_DIR):

        raise FileNotFoundError(
            f"\nValidation directory not found:\n"
            f"{VALIDATION_DIR}"
        )

    dataset = datasets.ImageFolder(
        VALIDATION_DIR,
        transform=get_inference_transform()
    )

    print("\nValidation dataset:")
    print("Images:", len(dataset))
    print("Classes:", dataset.classes)
    print("Class mapping:", dataset.class_to_idx)

    return dataset


# ============================================================
# COLLECT DETAILED PREDICTIONS
# ============================================================

@torch.no_grad()
def collect_predictions(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    records = []

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        sorted_probabilities, sorted_indices = torch.sort(
            probabilities,
            descending=True,
            dim=1
        )

        for i in range(images.size(0)):

            true_index = labels[i].item()

            top1_index = sorted_indices[i, 0].item()
            top2_index = sorted_indices[i, 1].item()

            top1_probability = (
                sorted_probabilities[i, 0].item()
            )

            top2_probability = (
                sorted_probabilities[i, 1].item()
            )

            margin = (
                top1_probability -
                top2_probability
            )

            correct = (
                top1_index ==
                true_index
            )

            records.append({
                "true_index": true_index,
                "predicted_index": top1_index,
                "true_class": CLASS_NAMES[true_index],
                "predicted_class": CLASS_NAMES[top1_index],
                "top1_confidence": top1_probability,
                "top2_confidence": top2_probability,
                "margin": margin,
                "correct": correct
            })

    return records


# ============================================================
# OVERALL ERROR ANALYSIS
# ============================================================

def overall_analysis(records):

    correct = [
        r for r in records
        if r["correct"]
    ]

    incorrect = [
        r for r in records
        if not r["correct"]
    ]

    print("\n" + "=" * 75)
    print("CORRECT vs INCORRECT PREDICTION ANALYSIS")
    print("=" * 75)

    print(
        f"Correct predictions   : "
        f"{len(correct)} / {len(records)}"
    )

    print(
        f"Incorrect predictions : "
        f"{len(incorrect)} / {len(records)}"
    )

    print("\nConfidence comparison:")

    if correct:
        print(
            f"Correct predictions:"
            f" mean={np.mean([r['top1_confidence'] for r in correct]) * 100:.2f}%"
            f", median={np.median([r['top1_confidence'] for r in correct]) * 100:.2f}%"
            f", min={np.min([r['top1_confidence'] for r in correct]) * 100:.2f}%"
            f", max={np.max([r['top1_confidence'] for r in correct]) * 100:.2f}%"
        )

    if incorrect:
        print(
            f"Incorrect predictions:"
            f" mean={np.mean([r['top1_confidence'] for r in incorrect]) * 100:.2f}%"
            f", median={np.median([r['top1_confidence'] for r in incorrect]) * 100:.2f}%"
            f", min={np.min([r['top1_confidence'] for r in incorrect]) * 100:.2f}%"
            f", max={np.max([r['top1_confidence'] for r in incorrect]) * 100:.2f}%"
        )

    print("\nMargin comparison:")

    if correct:
        print(
            f"Correct predictions:"
            f" mean={np.mean([r['margin'] for r in correct]) * 100:.2f}%"
            f", median={np.median([r['margin'] for r in correct]) * 100:.2f}%"
            f", min={np.min([r['margin'] for r in correct]) * 100:.2f}%"
            f", max={np.max([r['margin'] for r in correct]) * 100:.2f}%"
        )

    if incorrect:
        print(
            f"Incorrect predictions:"
            f" mean={np.mean([r['margin'] for r in incorrect]) * 100:.2f}%"
            f", median={np.median([r['margin'] for r in incorrect]) * 100:.2f}%"
            f", min={np.min([r['margin'] for r in incorrect]) * 100:.2f}%"
            f", max={np.max([r['margin'] for r in incorrect]) * 100:.2f}%"
        )


# ============================================================
# PER-CLASS ANALYSIS
# ============================================================

def per_class_analysis(records):

    print("\n" + "=" * 75)
    print("PER-CLASS ANALYSIS")
    print("=" * 75)

    for class_name in CLASS_NAMES:

        class_records = [
            r for r in records
            if r["true_class"] == class_name
        ]

        correct = [
            r for r in class_records
            if r["correct"]
        ]

        incorrect = [
            r for r in class_records
            if not r["correct"]
        ]

        print("\n" + class_name)
        print("-" * len(class_name))

        print(
            f"Total     : {len(class_records)}"
        )

        print(
            f"Correct   : {len(correct)}"
        )

        print(
            f"Incorrect : {len(incorrect)}"
        )

        if class_records:

            print(
                f"Accuracy  : "
                f"{len(correct) / len(class_records) * 100:.2f}%"
            )

            print(
                f"Mean confidence : "
                f"{np.mean([r['top1_confidence'] for r in class_records]) * 100:.2f}%"
            )

            print(
                f"Mean margin     : "
                f"{np.mean([r['margin'] for r in class_records]) * 100:.2f}%"
            )

        if incorrect:

            print("Most common wrong predictions:")

            wrong_predictions = {}

            for r in incorrect:

                predicted = r["predicted_class"]

                wrong_predictions[predicted] = (
                    wrong_predictions.get(
                        predicted,
                        0
                    ) + 1
                )

            sorted_errors = sorted(
                wrong_predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for predicted_class, count in sorted_errors:

                print(
                    f"  → {predicted_class}: "
                    f"{count}"
                )


# ============================================================
# MARGIN SAFETY ANALYSIS
# ============================================================

def margin_safety_analysis(records):

    print("\n" + "=" * 75)
    print("MARGIN SAFETY ANALYSIS")
    print("=" * 75)

    thresholds = [
        0.00,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.12,
        0.15
    ]

    print(
        f"{'Margin':<10}"
        f"{'Accepted':<12}"
        f"{'Rejected':<12}"
        f"{'Coverage':<12}"
        f"{'Accuracy':<12}"
        f"{'Errors':<10}"
    )

    print("-" * 68)

    labels = np.array([
        r["true_index"]
        for r in records
    ])

    predictions = np.array([
        r["predicted_index"]
        for r in records
    ])

    margins = np.array([
        r["margin"]
        for r in records
    ])

    for threshold in thresholds:

        accepted = margins >= threshold

        accepted_count = np.sum(accepted)
        rejected_count = len(records) - accepted_count

        if accepted_count > 0:

            accepted_accuracy = np.mean(
                predictions[accepted] ==
                labels[accepted]
            )

            errors = np.sum(
                predictions[accepted] !=
                labels[accepted]
            )

        else:

            accepted_accuracy = 0.0
            errors = 0

        coverage = (
            accepted_count /
            len(records)
        )

        print(
            f"{threshold * 100:>5.0f}%     "
            f"{accepted_count:<12}"
            f"{rejected_count:<12}"
            f"{coverage * 100:>6.2f}%     "
            f"{accepted_accuracy * 100:>7.2f}%     "
            f"{errors:<10}"
        )


# ============================================================
# LOW-MARGIN ERRORS
# ============================================================

def print_low_margin_errors(records):

    print("\n" + "=" * 75)
    print("LOW-MARGIN INCORRECT PREDICTIONS")
    print("=" * 75)

    incorrect = [
        r for r in records
        if not r["correct"]
    ]

    incorrect = sorted(
        incorrect,
        key=lambda r: r["margin"]
    )

    if not incorrect:

        print("No incorrect predictions found.")

        return

    for i, record in enumerate(
        incorrect[:20],
        start=1
    ):

        print(
            f"{i:02d}. "
            f"True={record['true_class']:<25} "
            f"Predicted={record['predicted_class']:<25} "
            f"Confidence={record['top1_confidence'] * 100:>6.2f}% "
            f"Margin={record['margin'] * 100:>6.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("ROBOCROP SOYBEAN DISEASE - ERROR ANALYSIS")
    print("=" * 75)

    model = load_model()

    dataset = load_validation_dataset()

    records = collect_predictions(
        model,
        dataset
    )

    overall_analysis(records)

    per_class_analysis(records)

    margin_safety_analysis(records)

    print_low_margin_errors(records)

    print("\n" + "=" * 75)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 75)

    print(
        "\nNOTE:"
        "\nOnly the validation set was used."
        "\nThe 140-image test set was not used for threshold selection."
    )


if __name__ == "__main__":
    main()