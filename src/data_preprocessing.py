from pathlib import Path
from collections import defaultdict
import pickle

from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CAPTION_FILE = DATA_DIR / "captions.txt"

MODEL_DIR = PROJECT_ROOT / "models"
TOKENIZER_FILE = MODEL_DIR / "tokenizer.pkl"


# --------------------------------------------------
# Load captions
# --------------------------------------------------

def load_captions(caption_file):
    """
    Load the Flickr8k captions file and create a mapping
    from image ID to its corresponding captions.
    """

    image_to_captions_mapping = defaultdict(list)

    with open(caption_file, "r", encoding="utf-8") as file:
        # Skip the header: image,caption
        next(file)

        captions_doc = file.read()

    for line in tqdm(captions_doc.split("\n"), desc="Loading captions"):

        # Split the line by comma
        tokens = line.split(",")

        if len(tokens) < 2:
            continue

        image_id, *captions = tokens

        # Remove image extension
        image_id = image_id.split(".")[0]

        # Convert captions list into a single string
        caption = " ".join(captions)

        image_to_captions_mapping[image_id].append(caption)

    return image_to_captions_mapping


# --------------------------------------------------
# Clean captions
# --------------------------------------------------

def clean_captions(image_to_captions_mapping):
    """
    Clean captions by:
    - converting text to lowercase
    - removing non-alphabetical characters
    - removing single-character words
    - adding startseq and endseq tokens
    """

    for image_id, captions in image_to_captions_mapping.items():

        for i in range(len(captions)):

            caption = captions[i]

            # Convert to lowercase
            caption = caption.lower()

            # Keep only alphabetic characters and spaces
            caption = "".join(
                char for char in caption
                if char.isalpha() or char.isspace()
            )

            # Remove extra spaces
            caption = " ".join(caption.split())

            # Add start and end tokens
            caption = (
                "startseq "
                + " ".join(
                    word for word in caption.split()
                    if len(word) > 1
                )
                + " endseq"
            )

            captions[i] = caption


# --------------------------------------------------
# Create tokenizer
# --------------------------------------------------

def create_tokenizer(image_to_captions_mapping):
    """
    Create a Keras tokenizer using all cleaned captions.
    """

    all_captions = [
        caption
        for captions in image_to_captions_mapping.values()
        for caption in captions
    ]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)

    return tokenizer, all_captions


# --------------------------------------------------
# Save tokenizer
# --------------------------------------------------

def save_tokenizer(tokenizer, tokenizer_file):
    """
    Save the trained tokenizer to disk.
    """

    tokenizer_file.parent.mkdir(parents=True, exist_ok=True)

    with open(tokenizer_file, "wb") as tokenizer_file_obj:
        pickle.dump(tokenizer, tokenizer_file_obj)


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

def load_tokenizer(tokenizer_file):
    """
    Load a previously saved tokenizer.
    """

    with open(tokenizer_file, "rb") as tokenizer_file_obj:
        tokenizer = pickle.load(tokenizer_file_obj)

    return tokenizer


# --------------------------------------------------
# Calculate vocabulary size and caption length
# --------------------------------------------------

def get_caption_statistics(tokenizer, all_captions):
    """
    Calculate:
    - vocabulary size
    - maximum caption length
    """

    max_caption_length = max(
        len(tokenizer.texts_to_sequences([caption])[0])
        for caption in all_captions
    )

    vocab_size = len(tokenizer.word_index) + 1

    return vocab_size, max_caption_length


# --------------------------------------------------
# Main preprocessing pipeline
# --------------------------------------------------

def preprocess_captions():
    """
    Complete caption preprocessing pipeline.
    """

    print("Loading captions...")

    image_to_captions_mapping = load_captions(CAPTION_FILE)

    total_captions = sum(
        len(captions)
        for captions in image_to_captions_mapping.values()
    )

    print(f"Total images: {len(image_to_captions_mapping)}")
    print(f"Total captions: {total_captions}")

    print("\nCleaning captions...")

    clean_captions(image_to_captions_mapping)

    print("\nCreating tokenizer...")

    tokenizer, all_captions = create_tokenizer(
        image_to_captions_mapping
    )

    vocab_size, max_caption_length = get_caption_statistics(
        tokenizer,
        all_captions
    )

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Maximum Caption Length: {max_caption_length}")

    print("\nSaving tokenizer...")

    save_tokenizer(tokenizer, TOKENIZER_FILE)

    print(f"Tokenizer saved to: {TOKENIZER_FILE}")

    return (
        image_to_captions_mapping,
        tokenizer,
        vocab_size,
        max_caption_length
    )


# --------------------------------------------------
# Run preprocessing
# --------------------------------------------------

if __name__ == "__main__":
    preprocess_captions()