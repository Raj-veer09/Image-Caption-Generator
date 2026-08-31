from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm

from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "Images"

MODEL_DIR = PROJECT_ROOT / "models"
FEATURE_FILE = MODEL_DIR / "efficientnet_features.pkl"


# --------------------------------------------------
# Build EfficientNet-B0 feature extractor
# --------------------------------------------------

def build_feature_extractor():
    """
    Load pretrained EfficientNet-B0 and remove the
    final classification layer.

    We keep the spatial feature map so that the
    attention mechanism can attend to different
    visual regions of the image.
    """

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False
    )

    # Freeze EfficientNet
    base_model.trainable = False

    return base_model


# --------------------------------------------------
# Extract features for one image
# --------------------------------------------------

def extract_single_image_feature(image_path, feature_model):
    """
    Extract spatial visual features from one image.
    """

    # EfficientNet-B0 expects 224 x 224 images
    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    # Convert image to NumPy array
    image = img_to_array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # EfficientNet preprocessing
    image = preprocess_input(image)

    # Extract spatial features
    feature = feature_model.predict(
        image,
        verbose=0
    )

    return feature


# --------------------------------------------------
# Extract features for all images
# --------------------------------------------------

def extract_features(image_dir, feature_model):
    """
    Extract EfficientNet spatial features for all
    images in the dataset.
    """

    image_features = {}

    image_files = [
        image_file
        for image_file in image_dir.iterdir()
        if image_file.is_file()
    ]

    print(f"Found {len(image_files)} images.")

    for image_file in tqdm(
        image_files,
        desc="Extracting EfficientNet features"
    ):

        feature = extract_single_image_feature(
            image_file,
            feature_model
        )

        # Remove .jpg extension
        image_id = image_file.stem

        image_features[image_id] = feature

    return image_features


# --------------------------------------------------
# Save features
# --------------------------------------------------

def save_features(image_features, feature_file):
    """
    Save extracted image features using pickle.
    """

    feature_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(feature_file, "wb") as file:
        pickle.dump(image_features, file)

    print(f"Features saved to: {feature_file}")


# --------------------------------------------------
# Load features
# --------------------------------------------------

def load_features(feature_file):
    """
    Load previously extracted image features.
    """

    with open(feature_file, "rb") as file:
        image_features = pickle.load(file)

    return image_features


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():

    print("Loading EfficientNet-B0...")

    feature_model = build_feature_extractor()

    print("EfficientNet-B0 feature extractor ready.")

    print("\nExtracting image features...")

    image_features = extract_features(
        IMAGE_DIR,
        feature_model
    )

    print(
        f"\nExtracted features for "
        f"{len(image_features)} images."
    )

    print("\nSaving features...")

    save_features(
        image_features,
        FEATURE_FILE
    )


# --------------------------------------------------
# Run script
# --------------------------------------------------

if __name__ == "__main__":
    main()