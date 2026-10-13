from predict import load_model, predict_image


IMAGE_PATH = r"..\test_images\IMG_20230917_144446.jpg"


print("=" * 70)
print("ROBOCROP SOYBEAN DISEASE TEST")
print("=" * 70)

print("Loading model...")

model = load_model()

print("Model loaded successfully.")
print()

print("Running prediction...")
print("Image:", IMAGE_PATH)
print()

result = predict_image(
    model,
    IMAGE_PATH
)


print("=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(
    "Top prediction:",
    result["disease"]
)

print(
    f"Top-1 confidence: "
    f"{result['confidence_percent']:.2f}%"
)

print(
    "Second prediction:",
    result["top2_disease"]
)

print(
    f"Top-2 confidence: "
    f"{result['top2_confidence_percent']:.2f}%"
)

print(
    f"Prediction margin: "
    f"{result['margin_percent']:.2f}%"
)

print()

print(
    "STATUS:",
    result["status"]
)

print(
    "ACTION:",
    result["action"]
)

print(
    "MESSAGE:",
    result["message"]
)


print()

print("=" * 70)
print("ALL CLASS PROBABILITIES")
print("=" * 70)

for prediction in result["top_predictions"]:

    print(
        f"{prediction['disease']:<28}"
        f"{prediction['confidence_percent']:>7.2f}%"
    )

print("=" * 70)