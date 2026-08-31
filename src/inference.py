from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from src.model import LuongDotProductAttention
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_caption_model.keras"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "tokenizer.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = load_model(
    MODEL_PATH,
    custom_objects={
        "LuongDotProductAttention": LuongDotProductAttention
    },
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

with open(
    TOKENIZER_PATH,
    "rb"
) as file:

    tokenizer = pickle.load(file)

print("Tokenizer loaded successfully.")


# ============================================================
# EFFICIENTNET FEATURE EXTRACTOR
# ============================================================

print("\nLoading EfficientNet-B0...")

from tensorflow.keras.applications import EfficientNetB0

feature_extractor = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling=None
)

print("EfficientNet-B0 loaded successfully.")


# ============================================================
# EXTRACT FEATURES FROM IMAGE
# ============================================================

def extract_features(image_path):
    """
    Extract EfficientNet-B0 spatial features.

    Input:
        Image

    Output:
        (49, 1280)
    """

    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image = img_to_array(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    image = preprocess_input(
        image
    )

    features = feature_extractor.predict(
        image,
        verbose=0
    )

    # Expected:
    # (1, 7, 7, 1280)

    features = np.squeeze(
        features,
        axis=0
    )

    # Convert:
    # (7, 7, 1280)
    #
    # to:
    # (49, 1280)

    features = features.reshape(
        49,
        1280
    )

    return features


# ============================================================
# WORD GENERATION
# ============================================================

def generate_caption(
    image_path,
    max_length=34
):
    """
    Generate a caption for one image.

    Uses greedy decoding:
    At every step, select the word
    with the highest probability.
    """

    # --------------------------------------------------------
    # Extract image features
    # --------------------------------------------------------

    features = extract_features(
        image_path
    )

    # Add batch dimension:
    #
    # (49, 1280)
    #     ↓
    # (1, 49, 1280)

    features = np.expand_dims(
        features,
        axis=0
    )


    # --------------------------------------------------------
    # Start caption
    # --------------------------------------------------------

    caption = "startseq"


    # --------------------------------------------------------
    # Generate one word at a time
    # --------------------------------------------------------

    for _ in range(
        max_length - 1
    ):

        # Convert current caption
        # into token IDs

        sequence = tokenizer.texts_to_sequences(
            [caption]
        )[0]


        # Predict next word

        padded_sequence = tf.keras.utils.pad_sequences(
            [sequence],
            maxlen=max_length,
            padding="post"
        )


        prediction = model.predict(
            [
                features,
                padded_sequence
            ],
            verbose=0
        )


        # Get highest-probability token

        predicted_id = int(
            np.argmax(
                prediction[0]
            )
        )


        # Convert ID back to word

        word = None

        for word_text, word_id in tokenizer.word_index.items():

            if word_id == predicted_id:

                word = word_text

                break


        # If token is unknown, stop

        if word is None:

            break


        # Stop at end token

        if word == "endseq":

            break


        # Add predicted word

        caption += " " + word


    # --------------------------------------------------------
    # Remove special token
    # --------------------------------------------------------

    caption = caption.replace(
        "startseq",
        ""
    ).strip()

    return caption


# ============================================================
# TEST IMAGE
# ============================================================

def main():

    image_path = (
        PROJECT_ROOT
        / "data"
        / "Images"
        / "1000268201_693b08cb0e.jpg"
    )

    print("\n")
    print("=" * 60)
    print("IMAGE CAPTION GENERATION")
    print("=" * 60)

    print(
        f"\nImage: {image_path}"
    )

    caption = generate_caption(
        image_path
    )

    print(
        "\nGenerated caption:"
    )

    print(
        caption
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

