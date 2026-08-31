from pathlib import Path
import pickle
import math

import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

CAPTION_FILE = DATA_DIR / "captions.txt"
TOKENIZER_FILE = MODEL_DIR / "tokenizer.pkl"
FEATURE_FILE = MODEL_DIR / "efficientnet_features.pkl"


# ============================================================
# LOAD AND CLEAN CAPTIONS
# ============================================================

def load_captions(caption_file):
    """
    Load captions from captions.txt.

    Returns:
        image_to_captions:
            {
                image_id: [
                    caption1,
                    caption2,
                    ...
                ]
            }
    """

    image_to_captions = {}

    with open(caption_file, "r", encoding="utf-8") as file:

        # Skip CSV header
        next(file)

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Split only at the first comma
            image_name, caption = line.split(",", 1)

            # Remove .jpg extension
            image_id = Path(image_name).stem

            # Convert to lowercase
            caption = caption.lower()

            # Keep only alphabetic characters and spaces
            caption = "".join(
                char
                for char in caption
                if char.isalpha() or char.isspace()
            )

            # Remove extra spaces
            caption = " ".join(caption.split())

            # Remove one-character words
            caption = " ".join(
                word
                for word in caption.split()
                if len(word) > 1
            )

            # Add special tokens
            caption = (
                "startseq "
                + caption
                + " endseq"
            )

            image_to_captions.setdefault(
                image_id,
                []
            ).append(caption)

    return image_to_captions


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer(tokenizer_file):
    """
    Load the tokenizer created during preprocessing.
    """

    with open(tokenizer_file, "rb") as file:
        tokenizer = pickle.load(file)

    return tokenizer


# ============================================================
# LOAD EFFICIENTNET FEATURES
# ============================================================

def load_image_features(feature_file):
    """
    Load EfficientNet-B0 features.

    Original feature shape:
        (1, 7, 7, 1280)

    Converted shape:
        (49, 1280)
    """

    with open(feature_file, "rb") as file:
        image_features = pickle.load(file)

    processed_features = {}

    for image_id, feature in image_features.items():

        feature = np.asarray(
            feature,
            dtype=np.float32
        )

        # Remove batch dimension
        # (1, 7, 7, 1280)
        #       ↓
        # (7, 7, 1280)

        feature = np.squeeze(
            feature,
            axis=0
        )

        # Preserve spatial locations
        #
        # (7, 7, 1280)
        #       ↓
        # (49, 1280)

        feature = feature.reshape(
            49,
            1280
        )

        processed_features[image_id] = feature

    return processed_features


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_image_ids(
    image_to_captions,
    train_ratio=0.8,
    val_ratio=0.1,
    random_seed=42
):
    """
    Split image IDs into train, validation and test sets.

    IMPORTANT:
    The split is performed at the IMAGE level.

    Therefore, all captions belonging to one image
    remain in the same split.

    Example:

        Image A
        ├── Caption 1
        ├── Caption 2
        ├── Caption 3
        ├── Caption 4
        └── Caption 5

    All five captions will belong to either
    train, validation or test.
    """

    image_ids = list(
        image_to_captions.keys()
    )

    rng = np.random.default_rng(
        random_seed
    )

    rng.shuffle(image_ids)

    total_images = len(image_ids)

    train_end = int(
        total_images * train_ratio
    )

    val_end = train_end + int(
        total_images * val_ratio
    )

    train_ids = image_ids[:train_end]

    val_ids = image_ids[
        train_end:val_end
    ]

    test_ids = image_ids[
        val_end:
    ]

    return (
        train_ids,
        val_ids,
        test_ids
    )


# ============================================================
# CREATE TRAINING SAMPLES
# ============================================================

def create_samples_for_images(
    image_ids,
    image_to_captions,
    tokenizer
):
    """
    Convert captions into input-target training samples.

    Example:

        Caption:
        startseq a dog is running endseq

        Creates:

        startseq
            -> a

        startseq a
            -> dog

        startseq a dog
            -> is

        startseq a dog is
            -> running

        startseq a dog is running
            -> endseq

    The image feature itself is NOT duplicated here.

    We only store:

        image_id
        input_sequence
        target_word
    """

    samples = []

    for image_id in image_ids:

        captions = image_to_captions[
            image_id
        ]

        for caption in captions:

            sequence = tokenizer.texts_to_sequences(
                [caption]
            )[0]

            for i in range(
                1,
                len(sequence)
            ):

                input_sequence = sequence[:i]

                target_word = sequence[i]

                samples.append(
                    (
                        image_id,
                        input_sequence,
                        target_word
                    )
                )

    return samples


# ============================================================
# MEMORY-EFFICIENT DATA GENERATOR
# ============================================================

class CaptionDataGenerator(Sequence):
    """
    Keras Sequence data generator.

    Only one batch is constructed at a time.

    This prevents us from creating a huge
    training array containing duplicated
    image features.
    """

    def __init__(
        self,
        samples,
        image_features,
        max_length,
        vocab_size,
        batch_size=32,
        shuffle=True
    ):

        self.samples = samples

        self.image_features = image_features

        self.max_length = max_length

        self.vocab_size = vocab_size

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indices = np.arange(
            len(self.samples)
        )

        self.on_epoch_end()


    # --------------------------------------------------------
    # Number of batches
    # --------------------------------------------------------

    def __len__(self):

        return math.ceil(
            len(self.samples)
            / self.batch_size
        )


    # --------------------------------------------------------
    # Generate one batch
    # --------------------------------------------------------

    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        current_batch_size = len(
            batch_indices
        )

        # ----------------------------------------------------
        # Image features
        #
        # Shape:
        # (batch, 49, 1280)
        # ----------------------------------------------------

        X_images = np.zeros(
            (
                current_batch_size,
                49,
                1280
            ),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Caption input
        #
        # Shape:
        # (batch, max_length)
        # ----------------------------------------------------

        X_sequences = np.zeros(
            (
                current_batch_size,
                self.max_length
            ),
            dtype=np.int32
        )

        # ----------------------------------------------------
        # Target word
        #
        # Shape:
        # (batch, vocab_size)
        # ----------------------------------------------------

        y = np.zeros(
            (
                current_batch_size,
                self.vocab_size
            ),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Fill batch
        # ----------------------------------------------------

        for i, sample_index in enumerate(
            batch_indices
        ):

            image_id, input_sequence, target_word = (
                self.samples[sample_index]
            )

            # Get visual features
            X_images[i] = self.image_features[
                image_id
            ]

            # Pad caption sequence
            X_sequences[i] = pad_sequences(
                [input_sequence],
                maxlen=self.max_length,
                padding="post"
            )[0]

            # One-hot target
            y[i] = to_categorical(
                target_word,
                num_classes=self.vocab_size
            )

        return (
            [
                X_images,
                X_sequences
            ],
            y
        )


    # --------------------------------------------------------
    # Shuffle after every epoch
    # --------------------------------------------------------

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# BUILD GENERATORS
# ============================================================

def build_data_generators(
    batch_size=32
):
    """
    Load all data and create:

        train_generator
        validation_generator
        test_generator

    The split is performed at the image level.
    """

    # --------------------------------------------------------
    # Load captions
    # --------------------------------------------------------

    print("Loading captions...")

    image_to_captions = load_captions(
        CAPTION_FILE
    )

    print(
        f"Loaded captions for "
        f"{len(image_to_captions)} images."
    )


    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = load_tokenizer(
        TOKENIZER_FILE
    )

    vocab_size = (
        len(tokenizer.word_index)
        + 1
    )

    print(
        f"Vocabulary size: "
        f"{vocab_size}"
    )


    # --------------------------------------------------------
    # Load EfficientNet features
    # --------------------------------------------------------

    print(
        "\nLoading EfficientNet features..."
    )

    image_features = load_image_features(
        FEATURE_FILE
    )

    print(
        f"Loaded features for "
        f"{len(image_features)} images."
    )


    # --------------------------------------------------------
    # Maximum caption length
    # --------------------------------------------------------

    max_length = max(
        len(
            tokenizer.texts_to_sequences(
                [caption]
            )[0]
        )
        for captions in image_to_captions.values()
        for caption in captions
    )

    print(
        f"Maximum caption length: "
        f"{max_length}"
    )


    # --------------------------------------------------------
    # Split images
    # --------------------------------------------------------

    print(
        "\nSplitting images..."
    )

    (
        train_ids,
        val_ids,
        test_ids
    ) = split_image_ids(
        image_to_captions
    )

    print(
        f"Training images: "
        f"{len(train_ids)}"
    )

    print(
        f"Validation images: "
        f"{len(val_ids)}"
    )

    print(
        f"Test images: "
        f"{len(test_ids)}"
    )


    # --------------------------------------------------------
    # Create samples
    # --------------------------------------------------------

    print(
        "\nCreating training samples..."
    )

    train_samples = create_samples_for_images(
        train_ids,
        image_to_captions,
        tokenizer
    )

    print(
        f"Training samples: "
        f"{len(train_samples)}"
    )


    print(
        "\nCreating validation samples..."
    )

    val_samples = create_samples_for_images(
        val_ids,
        image_to_captions,
        tokenizer
    )

    print(
        f"Validation samples: "
        f"{len(val_samples)}"
    )


    print(
        "\nCreating test samples..."
    )

    test_samples = create_samples_for_images(
        test_ids,
        image_to_captions,
        tokenizer
    )

    print(
        f"Test samples: "
        f"{len(test_samples)}"
    )


    # --------------------------------------------------------
    # Create generators
    # --------------------------------------------------------

    train_generator = CaptionDataGenerator(
        samples=train_samples,
        image_features=image_features,
        max_length=max_length,
        vocab_size=vocab_size,
        batch_size=batch_size,
        shuffle=True
    )

    validation_generator = CaptionDataGenerator(
        samples=val_samples,
        image_features=image_features,
        max_length=max_length,
        vocab_size=vocab_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_generator = CaptionDataGenerator(
        samples=test_samples,
        image_features=image_features,
        max_length=max_length,
        vocab_size=vocab_size,
        batch_size=batch_size,
        shuffle=False
    )


    return (
        train_generator,
        validation_generator,
        test_generator,
        tokenizer,
        vocab_size,
        max_length
    )


# ============================================================
# TEST THE DATASET
# ============================================================

def main():

    (
        train_generator,
        validation_generator,
        test_generator,
        tokenizer,
        vocab_size,
        max_length
    ) = build_data_generators(
        batch_size=32
    )


    print(
        "\n========================================"
    )

    print(
        "DATASET GENERATORS CREATED SUCCESSFULLY"
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # Check training batch
    # --------------------------------------------------------

    X_train, y_train = train_generator[0]

    train_images, train_captions = X_train

    print(
        "\nTraining batch:"
    )

    print(
        f"Image features: "
        f"{train_images.shape}"
    )

    print(
        f"Caption sequences: "
        f"{train_captions.shape}"
    )

    print(
        f"Targets: "
        f"{y_train.shape}"
    )


    # --------------------------------------------------------
    # Check validation batch
    # --------------------------------------------------------

    X_val, y_val = validation_generator[0]

    val_images, val_captions = X_val

    print(
        "\nValidation batch:"
    )

    print(
        f"Image features: "
        f"{val_images.shape}"
    )

    print(
        f"Caption sequences: "
        f"{val_captions.shape}"
    )

    print(
        f"Targets: "
        f"{y_val.shape}"
    )


    # --------------------------------------------------------
    # Check test batch
    # --------------------------------------------------------

    X_test, y_test = test_generator[0]

    test_images, test_captions = X_test

    print(
        "\nTest batch:"
    )

    print(
        f"Image features: "
        f"{test_images.shape}"
    )

    print(
        f"Caption sequences: "
        f"{test_captions.shape}"
    )

    print(
        f"Targets: "
        f"{y_test.shape}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

