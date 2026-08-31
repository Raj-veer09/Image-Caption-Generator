# 🖼️ Image Caption Generator

An end-to-end deep learning application that automatically generates natural-language captions for images using **EfficientNet-B0, Bidirectional LSTM, Luong Dot-Product Attention, and Beam Search decoding**.

The project combines computer vision and natural language processing by extracting spatial visual features from an image and using the partial caption as language context to attend to relevant image regions while predicting the next word.

---

## 📌 Project Overview

Image captioning requires a model to understand both:

- **What is present in an image**
- **How to describe that information using natural language**

This project implements an attention-based image captioning architecture consisting of:

- **EfficientNet-B0** as the visual feature extractor
- **256-dimensional word embeddings**
- **Bidirectional LSTM** for caption representation
- **Luong Dot-Product Attention** for visual-language alignment
- **Dense neural decoder** for next-word prediction
- **Beam Search** for improved caption generation
- **Streamlit** for an interactive web interface

---

## 🏗️ Architecture

The complete model pipeline is:

```text
                         IMAGE
                           │
                           ▼
                   EfficientNet-B0
                           │
                    7 × 7 × 1280
                           │
                        Reshape
                           │
                     49 × 1280
                           │
                     Dense(256)
                           │
                      49 × 256
                           │
                           │
                           ▼
                    Luong Attention
                           ▲
                           │
                           │
Caption → Tokenization → Embedding(256)
                           │
                           ▼
                        BiLSTM
                    256 + 256 units
                           │
                           ▼
                          512
                           │
                      Dense(256)
                           │
                           ▼
                 256-D Attention Query
                           │
                           ▼
                    Context Vector
                         (256)
                           │
                Query + Context Vector
                           │
                           ▼
                          512
                           │
                      Dense(256)
                           │
                           ▼
                   Dense(Vocabulary)
                           │
                         Softmax
                           │
                           ▼
                 Next-Word Probabilities
                           │
                           ▼
                     Beam Search
                           │
                           ▼
                   Generated Caption
```

---

## 🖼️ Visual Encoder — EfficientNet-B0

Images are resized to:

```text
224 × 224 × 3
```

A pretrained **EfficientNet-B0** model is used without its final classification layer.

The extracted feature map has shape:

```text
7 × 7 × 1280
```

The `7 × 7` spatial grid represents **49 image regions**, with each region represented by a **1,280-dimensional feature vector**.

Therefore:

```text
7 × 7 × 1280
        ↓
49 × 1280
```

These visual features are projected using a Dense layer:

```text
49 × 1280
        ↓
Dense(256)
        ↓
49 × 256
```

This gives the attention mechanism **49 visual feature vectors of 256 dimensions each**.

### Why EfficientNet-B0?

EfficientNet-B0 was selected because it provides:

- Strong pretrained ImageNet visual representations
- Good accuracy-to-computation trade-off
- Lower computational requirements than many larger CNN backbones
- Spatial `7 × 7` features suitable for attention
- Efficient feature extraction for CPU-based development

---

## 📝 Caption Preprocessing

Each image in the dataset contains multiple human-written captions.

The captions are cleaned and normalized before training.

The preprocessing pipeline includes:

```text
Raw Caption
     ↓
Convert to lowercase
     ↓
Remove unwanted punctuation / characters
     ↓
Clean tokens
     ↓
Add startseq
     ↓
Add endseq
     ↓
Tokenization
     ↓
Sequence generation
     ↓
Padding
```

Example:

```text
A little girl is playing with a dog.
```

becomes approximately:

```text
startseq little girl is playing with a dog endseq
```

The tokenizer maps each word to an integer ID.

### Caption Configuration

| Parameter | Value |
|---|---:|
| Vocabulary Size | 8,768 |
| Maximum Caption Length | 34 |
| Embedding Dimension | 256 |

Each token is transformed into a learned **256-dimensional embedding vector** before entering the BiLSTM.

---

## 🔄 Training Sample Generation

The model is trained for **next-word prediction**.

For a caption such as:

```text
startseq dog is running in park endseq
```

multiple training samples are created:

```text
Input                            Target

startseq                         dog
startseq dog                     is
startseq dog is                  running
startseq dog is running          in
startseq dog is running in       park
startseq dog is running in park  endseq
```

This allows the model to learn:

> Given the image and the correct caption context so far, predict the next word.

The training procedure therefore uses **teacher forcing**, because ground-truth previous words are provided as the caption prefix during training.

---

## 🔁 Bidirectional LSTM

After embedding, a caption has the representation:

```text
34 × 256
```

The language encoder uses a Bidirectional LSTM with:

```text
Forward LSTM  = 256 units
Backward LSTM = 256 units
```

The two directions are combined:

```text
256 + 256 = 512
```

The final forward and backward hidden states therefore form a:

```text
512-dimensional language representation
```

A Dense layer projects this representation into the shared attention dimension:

```text
512
 ↓
Dense(256)
 ↓
256-dimensional attention query
```

### Why BiLSTM instead of a single LSTM?

A normal LSTM processes the sequence in one direction.

A BiLSTM processes it in both:

```text
Forward:   startseq → dog → is → running

Backward:  running → is → dog → startseq
```

This provides a richer representation of the supplied caption sequence by incorporating contextual information from both directions.

---

## 🎯 Luong Dot-Product Attention

The attention mechanism determines which visual regions are most relevant to the current language representation.

After projection:

```text
Visual features = 49 × 256
Language query  =      256
```

The query is compared with each of the 49 visual vectors using a dot product:

```text
Query · Visual Vector 1 → Score 1
Query · Visual Vector 2 → Score 2
...
Query · Visual Vector 49 → Score 49
```

Softmax converts the scores into **49 attention weights**.

A weighted combination of the visual vectors produces the:

```text
256-dimensional context vector
```

This allows the model to emphasize different visual regions depending on the caption context.

---

## 🧠 Decoder

The model combines:

```text
Attention Query = 256
Context Vector  = 256
```

giving:

```text
256 + 256 = 512
```

The decoder then performs:

```text
512
 ↓
Dense(256)
 ↓
Dense(8768)
 ↓
Softmax
```

The final Softmax layer produces a probability distribution across the complete **8,768-word vocabulary**.

The selected decoding strategy then determines how those probabilities are used to construct the final caption.

---

## 🔍 Greedy Search vs Beam Search

Two decoding strategies were evaluated.

### Greedy Search

Greedy decoding selects the highest-probability word at every generation step.

```text
Current Caption
      ↓
Model Prediction
      ↓
Highest Probability Word
      ↓
Add Word
      ↓
Repeat
```

It is computationally efficient but may select a locally optimal word that leads to a weaker complete caption.

### Beam Search

Beam search retains multiple promising caption sequences simultaneously.

This project uses:

```text
Beam Width = 3
```

Instead of keeping only one candidate at each step, the decoder keeps the three strongest candidate sequences and expands them during subsequent predictions.

This provides a better approximation of the most probable complete caption.

---

## 📊 Model Evaluation

The model was evaluated using **BLEU-1 through BLEU-4**.

### Greedy vs Beam Search

| Metric | Greedy Search | Beam Search |
|---|---:|---:|
| BLEU-1 | 0.5244 | **0.5301** |
| BLEU-2 | 0.3410 | **0.3547** |
| BLEU-3 | 0.2125 | **0.2268** |
| BLEU-4 | 0.1293 | **0.1394** |

Beam search improved all four BLEU scores.

In particular, BLEU-4 increased from:

```text
0.1293 → 0.1394
```

which is approximately a **7.8% relative improvement**.

Therefore, Beam Search was selected as the final decoding strategy for the Streamlit application.

---

## 📈 Training Details

The dataset was divided at the **image level** to avoid captions belonging to the same image leaking across training and evaluation sets.

| Split | Images |
|---|---:|
| Training | 6,472 |
| Validation | 809 |
| Test | 810 |
| **Total** | **8,091** |

Generated next-word prediction samples:

| Split | Samples |
|---|---:|
| Training | 330,390 |
| Validation | 41,341 |
| Test | 41,634 |

Training configuration:

| Parameter | Value |
|---|---:|
| Epochs | 20 |
| Batch Size | 32 |
| Initial Learning Rate | 0.0001 |
| Best Epoch | 17 |
| Best Validation Loss | 3.8181 |

The model used **Early Stopping** and **ReduceLROnPlateau** during training.

The best model was obtained around epoch 17.

---

## 📂 Dataset

The project uses the **Flickr8k image-caption dataset**.

The dataset contains:

```text
8,091 images
```

with approximately:

```text
5 captions per image
```

During training, the visual representation of an image remains the same for its different captions, while each caption provides a different natural-language description of the same visual content.

This exposes the model to multiple valid ways of describing an image.

The image files are intentionally excluded from this GitHub repository.

---

## 🌐 Streamlit Application

An interactive Streamlit application is included.

The application allows the user to:

1. Upload a JPG, JPEG, or PNG image
2. Extract visual features using EfficientNet-B0
3. Generate a caption using the trained attention-based model
4. Decode the caption using Beam Search

Run the application with:

```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

```text
Image_Caption_Generator/
│
├── app.py
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│
├── models/
│
├── results/
│
└── src/
    ├── data_preprocessing.py
    ├── dataset.py
    ├── feature_extraction.py
    ├── model.py
    ├── train.py
    ├── inference.py
    ├── evaluate.py
    ├── beam_search.py
    ├── compare_decoding.py
    └── test_model.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Image_Caption_Generator
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .icg
.icg\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .icg
source .icg/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset and model artifacts

The image dataset, extracted EfficientNet features, tokenizer, and trained model files are not committed to the repository because they are generated or large binary artifacts.

Prepare the required dataset/model files before running inference or the Streamlit application.

### 5. Start Streamlit

```bash
python -m streamlit run app.py
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNet-B0
- Bidirectional LSTM
- Luong Dot-Product Attention
- NumPy
- Streamlit
- NLTK BLEU evaluation
- Git / GitHub

---

## 🚀 Key Features

- End-to-end image caption generation
- Transfer learning with EfficientNet-B0
- Spatial visual feature extraction
- Bidirectional language modeling
- Visual attention using Luong dot-product attention
- Teacher-forced next-word training
- Greedy and Beam Search decoding
- BLEU-based quantitative evaluation
- Train/validation/test separation
- Interactive Streamlit interface

---

## 📌 Future Improvements

Possible extensions include:

- Fine-tuning portions of EfficientNet on the captioning dataset
- Experimenting with Bahdanau attention
- Using Transformer-based decoders
- Comparing CNN-LSTM architecture against Vision Transformer models
- Adding attention-map visualization
- Evaluating with additional captioning metrics such as METEOR, ROUGE-L, CIDEr, and SPICE
- Deploying the application as a hosted web service

---

## 📄 Summary

This project demonstrates a complete multimodal deep-learning pipeline combining computer vision and natural language processing.

The final system uses:

```text
EfficientNet-B0
      +
Bidirectional LSTM
      +
Luong Dot-Product Attention
      +
Beam Search
```

to generate natural-language descriptions for unseen images.

The project covers the complete machine-learning lifecycle from **data preprocessing and feature extraction to training, evaluation, decoding optimization, and interactive inference**.