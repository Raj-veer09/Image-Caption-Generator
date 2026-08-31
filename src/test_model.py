import numpy as np

from src.dataset import build_data_generator
from src.model import create_model


def main():

    print("Loading data generator...")

    generator = build_data_generator(
        batch_size=32,
        shuffle=False
    )

    print("\nCreating model...")

    model = create_model(
        vocab_size=8768,
        max_length=34
    )

    print("\nGetting first batch...")

    X, y = generator[0]

    image_features, caption_sequences = X

    print("\nInput shapes:")
    print(
        "Image features:",
        image_features.shape
    )
    print(
        "Caption sequences:",
        caption_sequences.shape
    )
    print(
        "Target:",
        y.shape
    )

    print("\nRunning forward pass...")

    predictions = model.predict(
        [
            image_features,
            caption_sequences
        ],
        verbose=1
    )

    print("\nPrediction shape:")
    print(predictions.shape)

    # Check probability distribution
    print(
        "\nProbability sum for first sample:",
        np.sum(predictions[0])
    )

    print("\nForward pass successful!")


if __name__ == "__main__":
    main()