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

MODEL_PATH = PROJECT_ROOT / "models" / "best_caption_model.keras"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "efficientnet_features.pkl"
CAPTIONS_PATH = PROJECT_ROOT / "data" / "captions.txt"

MAX_LENGTH = 34
BEAM_WIDTH = 3


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

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

index_word = {
    value: key
    for key, value in tokenizer.word_index.items()
}

print(
    f"Vocabulary size: {len(tokenizer.word_index) + 1}"
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading EfficientNet features...")

with open(FEATURES_PATH, "rb") as file:
    image_features = pickle.load(file)

print(
    f"Loaded features for {len(image_features)} images."
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

            image_to_captions.setdefault(
                image_id,
                []
            ).append(caption)

    return image_to_captions


print("\nLoading captions...")

captions = load_captions()

print(
    f"Loaded captions for {len(captions)} images."
)


# ============================================================
# SAME TEST SPLIT AS DATASET.PY
# ============================================================

def get_test_images():

    image_ids = sorted(
        image_features.keys()
    )

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
# PREPARE FEATURES
# ============================================================

def prepare_features(feature):

    # Stored format:
    #
    # (1, 7, 7, 1280)

    if feature.ndim == 4:

        feature = np.squeeze(
            feature,
            axis=0
        )

    # (7, 7, 1280)
    #
    # ↓
    #
    # (49, 1280)

    feature = feature.reshape(
        49,
        1280
    )

    # (49, 1280)
    #
    # ↓
    #
    # (1, 49, 1280)

    feature = np.expand_dims(
        feature,
        axis=0
    )

    return feature


# ============================================================
# GREEDY DECODING
# ============================================================

def generate_greedy(
    feature
):

    feature = prepare_features(
        feature
    )

    caption = [
        "startseq"
    ]

    for _ in range(
        MAX_LENGTH - 1
    ):

        sequence = (
            tokenizer
            .texts_to_sequences(
                [" ".join(caption)]
            )[0]
        )

        padded = tf.keras.utils.pad_sequences(
            [sequence],
            maxlen=MAX_LENGTH,
            padding="post"
        )

        prediction = model.predict(
            [
                feature,
                padded
            ],
            verbose=0
        )[0]

        predicted_id = int(
            np.argmax(
                prediction
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
# BEAM SEARCH
# ============================================================

def generate_beam(
    feature,
    beam_width=3
):

    feature = prepare_features(
        feature
    )

    beams = [
        (
            ["startseq"],
            0.0
        )
    ]

    for _ in range(
        MAX_LENGTH - 1
    ):

        candidates = []

        for caption, score in beams:

            if caption[-1] == "endseq":

                candidates.append(
                    (
                        caption,
                        score
                    )
                )

                continue

            sequence = (
                tokenizer
                .texts_to_sequences(
                    [" ".join(caption)]
                )[0]
            )

            padded = tf.keras.utils.pad_sequences(
                [sequence],
                maxlen=MAX_LENGTH,
                padding="post"
            )

            prediction = model.predict(
                [
                    feature,
                    padded
                ],
                verbose=0
            )[0]

            top_indices = np.argsort(
                prediction
            )[-beam_width:][::-1]

            for token_id in top_indices:

                word = index_word.get(
                    int(token_id)
                )

                if word is None:
                    continue

                probability = float(
                    prediction[token_id]
                )

                probability = max(
                    probability,
                    1e-10
                )

                new_score = (
                    score
                    + np.log(probability)
                )

                new_caption = (
                    caption
                    + [word]
                )

                candidates.append(
                    (
                        new_caption,
                        new_score
                    )
                )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        beams = candidates[
            :beam_width
        ]

        if all(
            caption[-1] == "endseq"
            for caption, _ in beams
        ):
            break

    completed = [
        (caption, score)
        for caption, score in beams
        if caption[-1] == "endseq"
    ]

    if completed:

        best = max(
            completed,
            key=lambda x: x[1]
        )[0]

    else:

        best = max(
            beams,
            key=lambda x: x[1]
        )[0]

    return [
        word
        for word in best
        if word not in (
            "startseq",
            "endseq"
        )
    ]


# ============================================================
# BLEU CALCULATION
# ============================================================

def calculate_bleu(
    references,
    hypotheses
):

    smoothing = (
        SmoothingFunction()
        .method1
    )

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

    return bleu1, bleu2, bleu3, bleu4


# ============================================================
# MAIN
# ============================================================

def evaluate():

    print("\n")
    print("=" * 60)
    print("GREEDY VS BEAM SEARCH")
    print("=" * 60)

    test_ids = get_test_images()

    print(
        f"\nTest images: {len(test_ids)}"
    )

    greedy_references = []
    greedy_hypotheses = []

    beam_references = []
    beam_hypotheses = []


    # --------------------------------------------------------
    # Evaluate every test image
    # --------------------------------------------------------

    for i, image_id in enumerate(
        test_ids,
        start=1
    ):

        feature = image_features[
            image_id
        ]

        reference = captions[
            image_id
        ]

        greedy_caption = generate_greedy(
            feature
        )

        beam_caption = generate_beam(
            feature,
            BEAM_WIDTH
        )


        greedy_references.append(
            reference
        )

        greedy_hypotheses.append(
            greedy_caption
        )

        beam_references.append(
            reference
        )

        beam_hypotheses.append(
            beam_caption
        )


        # Show first 5 examples

        if i <= 5:

            print("\n" + "-" * 60)

            print(
                f"Image: {image_id}"
            )

            print(
                "Reference:",
                " ".join(reference[0])
            )

            print(
                "Greedy:",
                " ".join(greedy_caption)
            )

            print(
                "Beam:",
                " ".join(beam_caption)
            )


        if i % 50 == 0:

            print(
                f"\nProcessed "
                f"{i}/{len(test_ids)} images"
            )


    # ========================================================
    # CALCULATE SCORES
    # ========================================================

    greedy_scores = calculate_bleu(
        greedy_references,
        greedy_hypotheses
    )

    beam_scores = calculate_bleu(
        beam_references,
        beam_hypotheses
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(
        "\n                 Greedy       Beam Search"
    )

    print(
        f"BLEU-1          {greedy_scores[0]:.4f}       "
        f"{beam_scores[0]:.4f}"
    )

    print(
        f"BLEU-2          {greedy_scores[1]:.4f}       "
        f"{beam_scores[1]:.4f}"
    )

    print(
        f"BLEU-3          {greedy_scores[2]:.4f}       "
        f"{beam_scores[2]:.4f}"
    )

    print(
        f"BLEU-4          {greedy_scores[3]:.4f}       "
        f"{beam_scores[3]:.4f}"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    evaluate()

