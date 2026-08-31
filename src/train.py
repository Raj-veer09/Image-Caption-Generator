from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)

from src.dataset import build_data_generators
from src.model import create_model


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

BEST_MODEL_PATH = (
    MODEL_DIR / "best_caption_model.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR / "final_caption_model.keras"
)

HISTORY_PATH = (
    RESULTS_DIR / "training_history.json"
)

LOG_PATH = (
    RESULTS_DIR / "training_log.csv"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-4


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train():

    print("=" * 60)
    print("IMAGE CAPTION GENERATOR - TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Load dataset generators
    # --------------------------------------------------------

    print("\nLoading dataset...\n")

    (
        train_generator,
        validation_generator,
        test_generator,
        tokenizer,
        vocab_size,
        max_length
    ) = build_data_generators(
        batch_size=BATCH_SIZE
    )


    print("\nDataset ready.")

    print(
        f"Training batches: "
        f"{len(train_generator)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_generator)}"
    )

    print(
        f"Test batches: "
        f"{len(test_generator)}"
    )

    print(
        f"Vocabulary size: "
        f"{vocab_size}"
    )

    print(
        f"Maximum caption length: "
        f"{max_length}"
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating model...\n")

    model = create_model(
        vocab_size=vocab_size,
        max_length=max_length
    )

    model.summary()


    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    checkpoint = ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1
    )


    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        restore_best_weights=True,
        verbose=1
    )


    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        mode="min",
        verbose=1
    )


    csv_logger = CSVLogger(
        str(LOG_PATH),
        append=True
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(
        f"Epochs: {EPOCHS}"
    )
    print(
        f"Batch size: {BATCH_SIZE}"
    )
    print(
        f"Learning rate: {LEARNING_RATE}"
    )
    print("=" * 60)
    print("\n")


    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr,
            csv_logger
        ],
        verbose=1
    )


    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    print("\nSaving final model...")

    model.save(
        str(FINAL_MODEL_PATH)
    )

    print(
        f"Final model saved to:\n"
        f"{str(FINAL_MODEL_PATH)}"
    )


    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=4
        )


    print(
        f"Training history saved to:\n"
        f"{HISTORY_PATH}"
    )


    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    best_epoch = (
        min(
            range(
                len(
                    history.history["val_loss"]
                )
            ),
            key=lambda i:
            history.history["val_loss"][i]
        )
        + 1
    )

    best_val_loss = min(
        history.history["val_loss"]
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    train()

