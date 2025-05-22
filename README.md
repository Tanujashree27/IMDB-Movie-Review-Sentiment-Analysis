
# 🎬 IMDB Movie Review Sentiment Analysis

This project uses **BERT (Bidirectional Encoder Representations from Transformers)** to perform binary sentiment classification (positive/negative) on IMDB movie reviews.

## 📁 Dataset

* Source: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* Format: CSV with `sentence` and `label` columns.

## 👨‍💻 Author

* **Name:** Chandan Behera
* **Registration No:** 2201020388
* **CRANES Reg No:** CL2025010601924778

---

## 🚀 Project Workflow

### 1. Data Preprocessing

* Load dataset using `pandas`.
* Visualize review lengths for padding strategy.
* Train/Validation/Test split (70:15:15).
* Augmentation using `nlpaug` and NLTK (optional).

### 2. Tokenization

* BERT tokenizer (`bert-large-uncased`) with padding/truncation to max length 17.
* Convert tokens and attention masks to PyTorch tensors

### 3. Model Architecture

* Base: `bert-large-uncased` from Hugging Face Transformers.
* Custom classifier:

  * Dense → ReLU → Dropout → Dense → LogSoftmax

### 4. Training Setup

* Optimizer: `AdamW`
* Loss Function: `NLLLoss` with class weights
* Epochs: 20
* Early stopping based on validation loss
* Save best model weights to `saved_weights.pt`

### 5. Evaluation

* Load best model
* Predict on test set
* Display classification report (Precision, Recall, F1-score)

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib scikit-learn torch transformers nlpaug nltk
```

---

## 📊 Model Performance

* Evaluation includes precision, recall, and F1-score for binary classification.
* Designed to handle imbalanced class distributions using weighted loss

---

## 📌 Highlights

* Implements a full deep learning pipeline using **PyTorch + Transformers**.
* Efficient handling of imbalanced datasets using class weights.
* Utilizes GPU (if available) for training acceleration.
* Includes data augmentation setup for NLP using synonyms (WordNet).

---

## 📎 File Structure

```
main_code.py          # Full training + evaluation code
sentiment_train.csv   # IMDB dataset (should be placed in /content/ when using Colab)
saved_weights.pt      # Saved model weights (after training)
```

---

## 📣 Note

* The code is optimized for Google Colab.
* Modify file paths as needed if running locally.


