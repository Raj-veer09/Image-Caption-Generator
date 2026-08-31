from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from src.model import LuongDotProductAttention


# ============================================================
# PATHS
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

FEATURES_PATH = (
    PROJECT_ROOT
    / "models"
    / "efficientnet_features.pkl"
)

CAPTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "captions.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_LENGTH = 34


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = load_model(
    MODEL_PATH,
    custom_objects={
        "LuongDotProductAttention":
            LuongDotProductAttention
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

print(
    f"Vocabulary size: "
    f"{len(tokenizer.word_index) + 1}"
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading EfficientNet features...")

with open(
    FEATURES_PATH,
    "rb"
) as file:

    image_features = pickle.load(file)

print(
    f"Loaded features for "
    f"{len(image_features)} images."
)


# ============================================================
# LOAD CAPTIONS
# ============================================================

def load_captions():

    image_to_captions = {}

    with open(
        CAPTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        next(file)

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            image_id = parts[0].split(".")[0]

            caption = ",".join(parts[1:])

            # Same cleaning used during preprocessing

            caption = caption.lower()

            caption = "".join(
                char
                for char in caption
                if char.isalpha()
                or char.isspace()
            )

            words = [
                word
                for word in caption.split()
                if len(word) > 1
            ]

            caption = (
                ["startseq"]
                + words
                + ["endseq"]
            )

            if image_id not in image_to_captions:

                image_to_captions[
                    image_id
                ] = []

            image_to_captions[
                image_id
            ].append(caption)

    return image_to_captions


print("\nLoading captions...")

captions = load_captions()

print(
    f"Loaded captions for "
    f"{len(captions)} images."
)


# ============================================================
# CREATE TEST SPLIT
# ============================================================

def get_test_images():

    image_ids = sorted(
        image_features.keys()
    )

    # Same 80/10/10 image-level split
    # used in dataset.py

    np.random.seed(42)

    image_ids = np.array(
        image_ids
    )

    np.random.shuffle(
        image_ids
    )

    total = len(image_ids)

    train_end = int(
        0.80 * total
    )

    validation_end = int(
        0.90 * total
    )

    test_ids = image_ids[
        validation_end:
    ]

    return list(test_ids)


# ============================================================
# ID → WORD
# ============================================================

index_word = {
    value: key
    for key, value
    in tokenizer.word_index.items()
}


# ============================================================
# GENERATE CAPTION
# ============================================================


def generate_caption(
    feature,
    max_length=MAX_LENGTH
):

    # --------------------------------------------------------
    # Convert stored EfficientNet feature to correct shape
    # --------------------------------------------------------

    # Stored feature is:
    # (1, 7, 7, 1280)
    #
    # Remove the first dimension:
    # (7, 7, 1280)

    if feature.ndim == 4:
        feature = np.squeeze(
            feature,
            axis=0
        )

    # Convert:
    #
    # (7, 7, 1280)
    #
    # to:
    #
    # (49, 1280)

    feature = feature.reshape(
        49,
        1280
    )

    # Add batch dimension:
    #
    # (49, 1280)
    #      ↓
    # (1, 49, 1280)

    feature = np.expand_dims(
        feature,
        axis=0
    )


    # --------------------------------------------------------
    # Start caption
    # --------------------------------------------------------

    caption = [
        "startseq"
    ]


    # --------------------------------------------------------
    # Generate one word at a time
    # --------------------------------------------------------

    for _ in range(
        max_length - 1
    ):

        sequence = (
            tokenizer
            .texts_to_sequences(
                [" ".join(caption)]
            )[0]
        )

        padded = tf.keras.utils.pad_sequences(
            [sequence],
            maxlen=max_length,
            padding="post"
        )

        prediction = model.predict(
            [
                feature,
                padded
            ],
            verbose=0
        )

        predicted_id = int(
            np.argmax(
                prediction[0]
            )
        )

        word = index_word.get(
            predicted_id
        )

        if word is None:
            break

        if word == "endseq":
            break

        caption.append(
            word
        )

    return caption[1:]



# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate():

    print("\n")
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    test_ids = get_test_images()

    print(
        f"\nTest images: "
        f"{len(test_ids)}"
    )

    references = []
    hypotheses = []

    smoothing = (
        SmoothingFunction()
        .method1
    )


    # --------------------------------------------------------
    # Generate captions
    # --------------------------------------------------------

    for i, image_id in enumerate(
        test_ids,
        start=1
    ):

        feature = image_features[
            image_id
        ]

        generated = generate_caption(
            feature
        )

        reference_captions = (
            captions[image_id]
        )

        references.append(
            reference_captions
        )

        hypotheses.append(
            generated
        )


        if i <= 10:

            print(
                f"\nImage: {image_id}"
            )

            print(
                "Reference:",
                " ".join(
                    reference_captions[0]
                )
            )

            print(
                "Generated:",
                " ".join(generated)
            )


        if i % 50 == 0:

            print(
                f"\nProcessed "
                f"{i}/{len(test_ids)} images"
            )


    # --------------------------------------------------------
    # BLEU scores
    # --------------------------------------------------------

    bleu1 = corpus_bleu(
        references,
        hypotheses,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothing
    )

    bleu2 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing
    )

    bleu3 = corpus_bleu(
        references,
        hypotheses,
        weights=(1/3, 1/3, 1/3, 0),
        smoothing_function=smoothing
    )

    bleu4 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BLEU RESULTS")
    print("=" * 60)

    print(
        f"BLEU-1: {bleu1:.4f}"
    )

    print(
        f"BLEU-2: {bleu2:.4f}"
    )

    print(
        f"BLEU-3: {bleu3:.4f}"
    )

    print(
        f"BLEU-4: {bleu4:.4f}"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    evaluate()

