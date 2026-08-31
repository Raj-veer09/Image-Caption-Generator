import tensorflow as tf

from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Embedding,
    Dropout,
    Bidirectional,
    Concatenate
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "image_caption_model.keras"


# ============================================================
# Luong Dot-Product Attention
# ============================================================

class LuongDotProductAttention(tf.keras.layers.Layer):
    """
    Luong dot-product attention.

    Visual features:
        (batch, 49, attention_dim)

    Query:
        (batch, attention_dim)

    Output:
        Context vector:
        (batch, attention_dim)

        Attention weights:
        (batch, 49, 1)
    """

    def call(self, inputs):

        visual_features, query = inputs

        # ----------------------------------------------------
        # Query shape:
        #
        # (batch, 256)
        #
        # Convert to:
        #
        # (batch, 1, 256)
        # ----------------------------------------------------

        query = tf.expand_dims(
            query,
            axis=1
        )

        # ----------------------------------------------------
        # Calculate dot-product attention scores
        #
        # Visual features:
        # (batch, 49, 256)
        #
        # Query:
        # (batch, 1, 256)
        #
        # Result:
        # (batch, 49, 1)
        # ----------------------------------------------------

        scores = tf.matmul(
            visual_features,
            query,
            transpose_b=True
        )

        scores = tf.squeeze(
            scores,
            axis=-1
        )

        # ----------------------------------------------------
        # Convert scores to attention weights
        #
        # (batch, 49)
        # ----------------------------------------------------

        attention_weights = tf.nn.softmax(
            scores,
            axis=-1
        )

        # ----------------------------------------------------
        # Add dimension:
        #
        # (batch, 49)
        #      ↓
        # (batch, 49, 1)
        # ----------------------------------------------------

        attention_weights = tf.expand_dims(
            attention_weights,
            axis=-1
        )

        # ----------------------------------------------------
        # Calculate context vector
        #
        # attention weights:
        # (batch, 49, 1)
        #
        # visual features:
        # (batch, 49, 256)
        #
        # weighted features:
        # (batch, 49, 256)
        #
        # reduce over 49 visual locations:
        #
        # (batch, 256)
        # ----------------------------------------------------

        context_vector = tf.reduce_sum(
            attention_weights * visual_features,
            axis=1
        )

        return context_vector, attention_weights


# ============================================================
# Create Image Captioning Model
# ============================================================

def create_model(
    vocab_size,
    max_length,
    visual_feature_dim=1280,
    attention_dim=256,
    embedding_dim=256,
    lstm_units=256
):
    """
    EfficientNet-B0 + BiLSTM + Luong Dot-Product Attention.

    Image input:
        (batch, 49, 1280)

    Caption input:
        (batch, max_length)

    Output:
        (batch, vocab_size)
    """

    # ========================================================
    # IMAGE / VISUAL BRANCH
    # ========================================================

    image_input = Input(
        shape=(49, visual_feature_dim),
        name="image_features"
    )

    # EfficientNet:
    #
    # (batch, 49, 1280)
    #
    # Project to attention dimension:
    #
    # (batch, 49, 256)

    visual_features = Dense(
        attention_dim,
        activation="relu",
        name="visual_projection"
    )(image_input)

    visual_features = Dropout(
        0.3,
        name="visual_dropout"
    )(visual_features)


    # ========================================================
    # CAPTION / LANGUAGE BRANCH
    # ========================================================

    caption_input = Input(
        shape=(max_length,),
        name="caption_input"
    )

    # Token IDs:
    #
    # (batch, 34)
    #
    # Embedding:
    #
    # (batch, 34, 256)

    embeddings = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="word_embedding"
    )(caption_input)

    embeddings = Dropout(
        0.3,
        name="embedding_dropout"
    )(embeddings)


    # ========================================================
    # BIDIRECTIONAL LSTM
    # ========================================================

    caption_bilstm = Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=False,
            return_state=True
        ),
        name="caption_bilstm"
    )

    (
        lstm_output,
        forward_h,
        forward_c,
        backward_h,
        backward_c
    ) = caption_bilstm(embeddings)

    # --------------------------------------------------------
    # The LSTM output is not used directly.
    #
    # We use the final hidden states from both directions.
    #
    # forward_h:
    # (batch, 256)
    #
    # backward_h:
    # (batch, 256)
    #
    # Concatenate:
    #
    # (batch, 512)
    # --------------------------------------------------------

    language_state = Concatenate(
        name="language_state"
    )([
        forward_h,
        backward_h
    ])


    # ========================================================
    # CREATE ATTENTION QUERY
    # ========================================================

    # Convert:
    #
    # (batch, 512)
    #
    # into:
    #
    # (batch, 256)

    query = Dense(
        attention_dim,
        activation="tanh",
        name="attention_query"
    )(language_state)


    # ========================================================
    # LUONG DOT-PRODUCT ATTENTION
    # ========================================================

    attention_layer = LuongDotProductAttention(
        name="luong_attention"
    )

    context_vector, attention_weights = attention_layer(
        [
            visual_features,
            query
        ]
    )

    # context_vector:
    #
    # (batch, 256)


    # ========================================================
    # DECODER
    # ========================================================

    # Combine:
    #
    # Context vector:
    # (batch, 256)
    #
    # Query:
    # (batch, 256)
    #
    # Combined:
    # (batch, 512)

    decoder_input = Concatenate(
        name="decoder_concat"
    )([
        context_vector,
        query
    ])


    # --------------------------------------------------------
    # Decoder dense layer
    # --------------------------------------------------------

    decoder = Dense(
        256,
        activation="relu",
        name="decoder_dense"
    )(decoder_input)

    decoder = Dropout(
        0.3,
        name="decoder_dropout"
    )(decoder)


    # ========================================================
    # NEXT-WORD PREDICTION
    # ========================================================

    output = Dense(
        vocab_size,
        activation="softmax",
        name="word_prediction"
    )(decoder)


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = Model(
        inputs=[
            image_input,
            caption_input
        ],
        outputs=output,
        name="EfficientNet_BiLSTM_Luong_Captioner"
    )


    # ========================================================
    # COMPILE
    # ========================================================

    model.compile(
        optimizer=Adam(
            learning_rate=1e-4
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Test Model
# ============================================================

def main():

    # Values obtained from our preprocessing
    vocab_size = 8768
    max_length = 34

    print("Creating model...")

    model = create_model(
        vocab_size=vocab_size,
        max_length=max_length
    )

    print("\nModel created successfully.\n")

    model.summary()

    # Create models directory if necessary
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save model
    model.save(
        MODEL_FILE
    )

    print(
        f"\nModel saved to: {MODEL_FILE}"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
    

