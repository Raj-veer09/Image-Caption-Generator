from pathlib import Path
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.model import LuongDotProductAttention


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

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
# CONFIGURATION
# ============================================================

MAX_LENGTH = 34
BEAM_WIDTH = 3


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_caption_model():

    model = load_model(
        MODEL_PATH,
        custom_objects={
            "LuongDotProductAttention":
                LuongDotProductAttention
        },
        compile=False
    )

    return model


# ============================================================
# LOAD TOKENIZER
# ============================================================

@st.cache_resource
def load_tokenizer():

    with open(
        TOKENIZER_PATH,
        "rb"
    ) as file:

        tokenizer = pickle.load(file)

    return tokenizer


# ============================================================
# LOAD EFFICIENTNET
# ============================================================

@st.cache_resource
def load_feature_extractor():

    model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling=None
    )

    return model


# ============================================================
# EXTRACT IMAGE FEATURES
# ============================================================

def extract_features(
    image,
    feature_extractor
):

    # Resize image
    image = image.resize(
        (224, 224)
    )

    # Convert to NumPy array
    image = img_to_array(
        image
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # EfficientNet preprocessing
    image = preprocess_input(
        image
    )

    # Extract features
    features = feature_extractor.predict(
        image,
        verbose=0
    )

    # EfficientNet output:
    # (1, 7, 7, 1280)

    features = np.squeeze(
        features,
        axis=0
    )

    # Convert:
    # (7, 7, 1280)
    # →
    # (49, 1280)

    features = features.reshape(
        49,
        1280
    )

    # Add batch dimension
    features = np.expand_dims(
        features,
        axis=0
    )

    # Final:
    # (1, 49, 1280)

    return features


# ============================================================
# BEAM SEARCH CAPTION GENERATION
# ============================================================

def generate_caption_beam_search(
    model,
    tokenizer,
    features,
    beam_width=3
):

    index_word = {
        value: key
        for key, value
        in tokenizer.word_index.items()
    }

    # Each beam contains:
    #
    # (caption_words, score)
    #
    # Start with only startseq.

    beams = [
        (
            ["startseq"],
            0.0
        )
    ]

    for _ in range(
        MAX_LENGTH - 1
    ):

        all_candidates = []

        # Expand every current beam

        for caption, score in beams:

            # If this caption already ended,
            # keep it unchanged.

            if caption[-1] == "endseq":

                all_candidates.append(
                    (
                        caption,
                        score
                    )
                )

                continue

            # Convert caption to token IDs

            sequence = (
                tokenizer
                .texts_to_sequences(
                    [" ".join(caption)]
                )[0]
            )

            # Pad to maximum caption length

            padded_sequence = (
                tf.keras.utils.pad_sequences(
                    [sequence],
                    maxlen=MAX_LENGTH,
                    padding="post"
                )
            )

            # Predict next-word probabilities

            prediction = model.predict(
                [
                    features,
                    padded_sequence
                ],
                verbose=0
            )[0]

            # Avoid log(0)

            prediction = np.maximum(
                prediction,
                1e-10
            )

            # Take the most promising words

            top_indices = np.argsort(
                prediction
            )[-beam_width:][::-1]

            # Create candidate beams

            for token_id in top_indices:

                word = index_word.get(
                    int(token_id)
                )

                if word is None:
                    continue

                # Add log probability to the
                # existing sequence score.

                candidate_score = (
                    score
                    + np.log(
                        prediction[
                            token_id
                        ]
                    )
                )

                candidate_caption = (
                    caption
                    + [word]
                )

                all_candidates.append(
                    (
                        candidate_caption,
                        candidate_score
                    )
                )

        # Keep only the best beam_width
        # candidates.

        beams = sorted(
            all_candidates,
            key=lambda x: x[1],
            reverse=True
        )[:beam_width]

        # Stop if all beams have ended

        if all(
            caption[-1] == "endseq"
            for caption, _ in beams
        ):
            break

    # Select the highest-scoring caption

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

    return " ".join(
        words
    )


# ============================================================
# LOAD COMPONENTS
# ============================================================

model = load_caption_model()

tokenizer = load_tokenizer()

feature_extractor = load_feature_extractor()


# ============================================================
# STREAMLIT UI
# ============================================================

st.title(
    "🖼️ Image Caption Generator"
)

st.write(
    "Upload an image and let the model generate a caption."
)

st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# GENERATE CAPTION
# ============================================================

if uploaded_file is not None:

    image = load_img(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Image"
    )

    st.divider()

    if st.button(
        "Generate Caption",
        type="primary"
    ):

        with st.spinner(
            "Generating caption..."
        ):

            # Extract EfficientNet features

            features = extract_features(
                image,
                feature_extractor
            )

            # Generate caption using
            # beam search

            caption = generate_caption_beam_search(
                model,
                tokenizer,
                features,
                beam_width=BEAM_WIDTH
            )

        st.subheader(
            "Generated Caption"
        )

        st.success(
            caption.capitalize() + "."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "About the model"
):

    st.write(
        """
        **Architecture**

        • EfficientNet-B0 visual encoder
        • BiLSTM language encoder
        • Luong dot-product attention
        • Dense decoder
        • 8,768-word vocabulary

        **Inference**

        • Beam Search
        • Beam width: 3

        **Evaluation**

        • Greedy BLEU-1: 0.5244
        • Greedy BLEU-2: 0.3410
        • Greedy BLEU-3: 0.2125
        • Greedy BLEU-4: 0.1293

        • Beam Search BLEU-1: 0.5301
        • Beam Search BLEU-2: 0.3547
        • Beam Search BLEU-3: 0.2268
        • Beam Search BLEU-4: 0.1394
        """
    )
    
