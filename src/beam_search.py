from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

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

MAX_LENGTH = 34


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model(
    MODEL_PATH,
    custom_objects={
        "LuongDotProductAttention":
            LuongDotProductAttention
    },
    compile=False
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

with open(
    TOKENIZER_PATH,
    "rb"
) as file:

    tokenizer = pickle.load(file)


index_word = {
    value: key
    for key, value
    in tokenizer.word_index.items()
}


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(feature):

    # Stored EfficientNet feature:
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
# BEAM SEARCH
# ============================================================

def generate_caption_beam_search(
    feature,
    beam_width=3,
    max_length=MAX_LENGTH
):

    feature = prepare_features(
        feature
    )

    # Each beam contains:
    #
    # (caption, score)

    beams = [
        (
            ["startseq"],
            0.0
        )
    ]


    # --------------------------------------------------------
    # Generate one token at a time
    # --------------------------------------------------------

    for _ in range(
        max_length - 1
    ):

        candidates = []


        # ----------------------------------------------------
        # Expand every current beam
        # ----------------------------------------------------

        for caption, score in beams:

            # If already finished,
            # keep it unchanged

            if caption[-1] == "endseq":

                candidates.append(
                    (
                        caption,
                        score
                    )
                )

                continue


            # Convert caption to IDs

            sequence = (
                tokenizer
                .texts_to_sequences(
                    [" ".join(caption)]
                )[0]
            )


            # Pad sequence

            padded = (
                tf.keras.utils
                .pad_sequences(
                    [sequence],
                    maxlen=max_length,
                    padding="post"
                )
            )


            # Model prediction

            prediction = model.predict(
                [
                    feature,
                    padded
                ],
                verbose=0
            )[0]


            # ------------------------------------------------
            # Select top beam_width words
            # ------------------------------------------------

            top_indices = np.argsort(
                prediction
            )[-beam_width:][::-1]


            for token_id in top_indices:

                word = index_word.get(
                    int(token_id)
                )

                if word is None:
                    continue


                # Use log probability
                # to avoid multiplying
                # many small probabilities

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


        # ----------------------------------------------------
        # Keep best beam_width candidates
        # ----------------------------------------------------

        candidates = sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True
        )


        beams = candidates[
            :beam_width
        ]


        # ----------------------------------------------------
        # Stop if every beam finished
        # ----------------------------------------------------

        if all(
            caption[-1] == "endseq"
            for caption, _ in beams
        ):

            break


    # ========================================================
    # Select best completed caption
    # ========================================================

    completed = [
        (caption, score)
        for caption, score
        in beams
        if caption[-1] == "endseq"
    ]


    if completed:

        best_caption = max(
            completed,
            key=lambda x: x[1]
        )[0]

    else:

        best_caption = max(
            beams,
            key=lambda x: x[1]
        )[0]


    # Remove special tokens

    words = [
        word
        for word in best_caption
        if word not in (
            "startseq",
            "endseq"
        )
    ]

    return " ".join(words)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    FEATURES_PATH = (
        PROJECT_ROOT
        / "models"
        / "efficientnet_features.pkl"
    )

    with open(
        FEATURES_PATH,
        "rb"
    ) as file:

        features = pickle.load(
            file
        )


    image_id = (
        "1000268201_693b08cb0e"
    )

    feature = features[
        image_id
    ]


    print("=" * 60)
    print("BEAM SEARCH CAPTION GENERATION")
    print("=" * 60)

    caption = generate_caption_beam_search(
        feature,
        beam_width=3
    )

    print(
        "\nGenerated caption:"
    )

    print(
        caption
    )

    print("=" * 60)

