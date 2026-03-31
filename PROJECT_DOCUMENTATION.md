# 📊 Project Documentation
## Financial Text Sentiment Analysis with Sarcasm Detection & Explainable AI

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Architecture](#2-project-architecture)
3. [File Structure & Descriptions](#3-file-structure--descriptions)
4. [Libraries Used — Detailed Explanation](#4-libraries-used--detailed-explanation)
5. [Phase 1: Baseline Model (TF-IDF + Logistic Regression)](#5-phase-1-baseline-model)
6. [Phase 2: DistilBERT Sentiment Model](#6-phase-2-distilbert-sentiment-model)
7. [Phase 3: DistilBERT Sarcasm Detection](#7-phase-3-distilbert-sarcasm-detection)
8. [Phase 4: Explainable AI (SHAP)](#8-phase-4-explainable-ai-shap)
9. [Dashboard (Streamlit)](#9-dashboard-streamlit)
10. [Data Pipeline & Preprocessing](#10-data-pipeline--preprocessing)
11. [Results & Findings](#11-results--findings)
12. [How to Run the Project](#12-how-to-run-the-project)

---

## 1. Project Overview

### 1.1 Problem Statement

Financial text from social media (Reddit's WallStreetBets) contains significant amounts of **sarcasm**, which causes traditional sentiment classifiers to misclassify text. For example:

- *"Great job losing all my money, really smart move"* → Surface sentiment is **positive** (words like "great", "smart") but the actual sentiment is **negative**.
- *"My portfolio is doing amazing, only down 50%"* → Surface sentiment is **positive** but actual sentiment is **negative**.

This project solves this problem by building a **multi-model sentiment analysis pipeline** that:
1. Establishes a traditional ML baseline (TF-IDF + Logistic Regression)
2. Fine-tunes a **DistilBERT transformer** model for sentiment classification
3. Fine-tunes a separate **DistilBERT model** for sarcasm detection
4. Corrects sentiment predictions based on sarcasm detection (polarity flipping)
5. Provides **Explainable AI (SHAP)** to understand which words influence predictions
6. Compares traditional ML vs deep learning approaches

### 1.2 Dataset

| Property | Value |
|----------|-------|
| **Source** | Reddit WallStreetBets (r/wallstreetbets) |
| **Rows** | ~1,000,000 posts |
| **File Size** | ~146 MB (CSV) |
| **Key Column** | `title` (post titles used as input text) |
| **Other Columns** | objectid, score, author, author_flair_text, removed_by, total_awards_received, awarders, created_utc, full_link, num_comments, over_18 |
| **Labels** | Simulated (Sentiment: Positive/Negative/Neutral, Sarcasm: 0/1) |

### 1.3 Models & Approaches

| Model | Type | Purpose | Training Data |
|-------|------|---------|---------------|
| **Baseline (TF-IDF + LR)** | Traditional ML | Sentiment classification | Full dataset (~1M rows) |
| **DistilBERT Sentiment** | Transformer (Fine-tuned) | Sentiment classification | 10K rows (1 epoch) |
| **DistilBERT Sarcasm** | Transformer (Fine-tuned) | Sarcasm detection (binary) | 10K rows (1 epoch) |

---

## 2. Project Architecture

```
┌─────────────────────────────────────────────────────┐
│                 RAW DATA (1M rows)                  │
│              r_wallstreetbets_big.csv                │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              PREPROCESSING (utils.py)               │
│  • Lowercase conversion                             │
│  • URL removal                                      │
│  • Reddit mention removal (u/username)              │
│  • Special character removal                        │
│  • Multiple space normalization                     │
│  • DistilBertDataset (PyTorch Dataset class)        │
└────────────────────┬────────────────────────────────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│  PHASE 1     │ │  PHASE 2     │ │  PHASE 3         │
│  Baseline    │ │  DistilBERT  │ │  DistilBERT      │
│  TF-IDF +    │ │  Sentiment   │ │  Sarcasm         │
│  Logistic    │ │  Fine-tuned  │ │  Fine-tuned      │
│  Regression  │ │  3-class     │ │  Binary (0/1)    │
│              │ │  classifier  │ │  classifier      │
│  ~1M rows    │ │  10K rows    │ │  10K rows        │
│  50K TF-IDF  │ │  128 tokens  │ │  128 tokens      │
│  features    │ │  1 epoch     │ │  1 epoch         │
└──────┬───────┘ └──────┬───────┘ └────────┬─────────┘
       │                │                   │
       └────────────────┼───────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│        ACCURACY COMPARISON (app.py)                 │
│  Baseline vs DistilBERT — side-by-side metrics      │
│  Confusion matrices, classification reports          │
│  Training time comparison                            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         LIVE TEXT PREDICTION (app.py)                │
│  Text → Sarcasm Model → Is Sarcastic?               │
│    YES: Flip sentiment polarity (Pos↔Neg)            │
│    NO:  Keep original sentiment                      │
│  → Final sarcasm-aware sentiment prediction          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           INTERACTIVE DASHBOARD (app.py)            │
│  Streamlit + Plotly + streamlit-option-menu          │
│  7 Pages: Home, Dataset, Baseline, Sarcasm,         │
│           Comparison, Live Prediction, Explainability│
└─────────────────────────────────────────────────────┘
```

---

## 3. File Structure & Descriptions

```
palak/
│
├── app.py                         ← Streamlit dashboard (7 interactive pages)
├── train_baseline.py              ← Phase 1: Baseline sentiment model (TF-IDF + LR)
├── train_distilbert_sentiment.py  ← Phase 2: DistilBERT sentiment fine-tuning
├── train_distilbert_sarcasm.py    ← Phase 3: DistilBERT sarcasm fine-tuning
├── train_sarcasm.py               ← Legacy: Old sarcasm pipeline (TF-IDF based)
├── utils.py                       ← Shared utilities (preprocessing, dataset class)
├── generate_data.py               ← Synthetic dataset generator (1M rows)
├── requirements.txt               ← Python dependencies
├── r_wallstreetbets_big.csv       ← Dataset (1M rows, ~146MB)
├── PROJECT_DOCUMENTATION.md       ← This documentation file
│
└── models/                        ← Saved model artifacts
    ├── baseline_model.joblib            ← Trained Logistic Regression (Phase 1)
    ├── tfidf_vectorizer.joblib          ← Fitted TF-IDF vectorizer (50K features)
    ├── label_encoder.joblib             ← Label encoder (Positive/Negative/Neutral)
    ├── test_data.joblib                 ← Saved test set for SHAP computation
    ├── baseline_metrics.joblib          ← Phase 1 results (accuracy, confusion matrix)
    │
    ├── distilbert_sentiment/            ← DistilBERT Sentiment Model (Phase 2)
    │   ├── config.json                  ← Model configuration
    │   ├── model.safetensors            ← Model weights
    │   ├── tokenizer_config.json        ← Tokenizer config
    │   ├── vocab.txt                    ← DistilBERT vocabulary
    │   ├── sentiment_metrics.joblib     ← Sentiment model metrics
    │   └── label_encoder.joblib         ← Sentiment label encoder
    │
    └── distilbert_sarcasm/              ← DistilBERT Sarcasm Model (Phase 3)
        ├── config.json                  ← Model configuration
        ├── model.safetensors            ← Model weights
        ├── tokenizer_config.json        ← Tokenizer config
        ├── vocab.txt                    ← DistilBERT vocabulary
        └── sarcasm_metrics.joblib       ← Sarcasm model metrics
```

### File-by-File Explanation

#### `utils.py` (~479 lines)
**Purpose:** Shared utility module containing all reusable functions.
- **Text Preprocessing:** `preprocess_text()` — cleans raw text (lowercase, remove URLs, mentions, special chars)
- **Batch Processing:** `preprocess_batch()` — applies preprocessing to entire pandas Series efficiently
- **Label Simulation:** `simulate_sentiment_labels_batch()` — creates realistic sentiment labels using keyword heuristics
- **Sarcasm Simulation:** `simulate_sarcasm_labels_batch()` — simulates sarcasm labels using caps ratio, exclamation marks, and mixed-signal keywords
- **Data Loading:** `load_data_chunked()` — memory-efficient CSV loading in chunks for 1M+ rows
- **Sarcasm Adjustment:** `flip_sentiment()` and `adjust_sentiments_for_sarcasm()` — flip polarity for sarcastic texts
- **DistilBertDataset:** PyTorch `Dataset` class for tokenizing text and creating DataLoader-compatible batches

#### `train_baseline.py` (~314 lines)
**Purpose:** Train the baseline sentiment classifier using traditional ML (Phase 1).
- Loads full dataset (~1M rows), preprocesses text, simulates labels
- Splits data 80/20 with stratification
- Fits TF-IDF vectorizer (50,000 features, unigrams + bigrams)
- Trains Logistic Regression with SAGA solver
- Saves all artifacts to `models/` directory

#### `train_distilbert_sentiment.py` (~243 lines)
**Purpose:** Fine-tune DistilBERT for sentiment classification (Phase 2).
- Loads 10,000 rows for efficient training
- Tokenizes text using `DistilBertTokenizer` (max_length=128)
- Fine-tunes `DistilBertForSequenceClassification` with 3 output labels
- Uses AdamW optimizer with linear learning rate scheduling
- Trains for 1 epoch, saves best model based on validation accuracy

#### `train_distilbert_sarcasm.py` (~236 lines)
**Purpose:** Fine-tune DistilBERT for sarcasm detection (Phase 3).
- Loads 10,000 rows, simulates both sentiment and sarcasm labels
- Fine-tunes `DistilBertForSequenceClassification` with 2 output labels (sarcastic/not)
- Same training setup as sentiment model (AdamW, linear scheduling)
- Saves model and metrics to `models/distilbert_sarcasm/`

#### `app.py` (~1000+ lines)
**Purpose:** Interactive Streamlit dashboard with 7 pages.
- Uses Plotly for all charts (dark-themed, interactive)
- Uses `streamlit-option-menu` for sidebar navigation
- Custom CSS for gradient metric cards, dark tables, and styled sections
- Live text prediction pipeline using both DistilBERT models
- Accuracy comparison between Baseline and DistilBERT models

---

## 4. Libraries Used — Detailed Explanation

### 4.1 Core Data Science Libraries

#### **Pandas** (`pandas>=1.5.0`)
- **What:** Python library for data manipulation and analysis. Provides DataFrame — a 2D tabular data structure.
- **Why Used:** To load, process, and manipulate the 1M-row CSV dataset efficiently.
- **Where Used:** `utils.py`, `train_baseline.py`, `train_distilbert_sentiment.py`, `train_distilbert_sarcasm.py`, `app.py`

#### **NumPy** (`numpy>=1.23.0`)
- **What:** Fundamental library for numerical computing. Provides fast N-dimensional arrays.
- **Why Used:** Array operations, metric calculations, random number generation.
- **Where Used:** All training files and `app.py`

#### **SciPy** (`scipy>=1.10.0`)
- **What:** Scientific computing library built on NumPy. Provides sparse matrices.
- **Why Used:** Sparse matrix handling for TF-IDF features in the baseline model.

---

### 4.2 Machine Learning Libraries

#### **Scikit-learn** (`scikit-learn>=1.2.0`)
- **What:** Widely used ML library providing preprocessing, classification, and evaluation tools.
- **Where Used:**

| Component | Class/Function | Purpose |
|-----------|---------------|---------|
| **TF-IDF Vectorizer** | `TfidfVectorizer` | Convert text to numerical features (baseline) |
| **Logistic Regression** | `LogisticRegression` | Baseline classification model |
| **Label Encoder** | `LabelEncoder` | Convert text labels to numbers |
| **Train/Test Split** | `train_test_split` | Split data 80/20 |
| **Metrics** | `accuracy_score`, `precision_score`, `recall_score`, `f1_score` | Evaluate models |
| **Reports** | `confusion_matrix`, `classification_report` | Generate detailed results |

##### TF-IDF Vectorizer Configuration (Baseline)
```python
TfidfVectorizer(
    max_features=50000,       # Top 50,000 words
    ngram_range=(1, 2),       # Unigrams + bigrams
    sublinear_tf=True,        # Log normalization
    min_df=5,                 # Min document frequency
    max_df=0.95,              # Max document frequency
    dtype=np.float32          # Memory-efficient
)
```

##### Logistic Regression Configuration (Baseline)
```python
LogisticRegression(
    max_iter=1000,        # Max iterations
    solver='saga',        # Fast for large datasets
    C=1.0,                # Regularization
    n_jobs=-1,            # Use all CPU cores
    random_state=42       # Reproducibility
)
```

---

### 4.3 Deep Learning Libraries

#### **PyTorch** (`torch>=2.0.0`)
- **What:** Deep learning framework providing tensor computation and automatic differentiation.
- **Why Used:** Required for fine-tuning DistilBERT transformer models.
- **Where Used:**
  - `train_distilbert_sentiment.py` — Model training, optimizer, GPU acceleration
  - `train_distilbert_sarcasm.py` — Same training pipeline for sarcasm
  - `utils.py` — `DistilBertDataset` (PyTorch `Dataset` subclass)
  - `app.py` — Model inference for live predictions

#### **Transformers** (`transformers>=4.30.0`) — Hugging Face
- **What:** Library providing pre-trained transformer models (BERT, GPT, DistilBERT, etc.).
- **Why Used:** Provides `DistilBertTokenizer` and `DistilBertForSequenceClassification` for fine-tuning on financial text.
- **Key Components:**

| Component | Purpose |
|-----------|---------|
| `DistilBertTokenizer` | Tokenize text into WordPiece subwords with attention masks |
| `DistilBertForSequenceClassification` | Pre-trained DistilBERT with classification head |
| `get_linear_schedule_with_warmup` | Learning rate scheduler for gradual warmup |

##### DistilBERT Configuration
```python
# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# max_length=128, padding='max_length', truncation=True

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3  # Positive/Negative/Neutral (Sentiment)
    # num_labels=2  # Sarcastic/Not Sarcastic (Sarcasm)
)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
```

##### Why DistilBERT?
- **66M parameters** (vs 110M for BERT-base) — 40% smaller
- **60% faster inference** than BERT with 97% of BERT's performance
- **Pre-trained on English Wikipedia + BookCorpus** — rich language understanding
- **Contextual embeddings** — understands word meaning from context (unlike TF-IDF bag-of-words)
- **Transfer learning** — pre-trained knowledge transferred to financial text domain

#### **tqdm** (`tqdm>=4.65.0`)
- **What:** Progress bar library for loops and iterables.
- **Why Used:** Shows training/evaluation progress during DistilBERT fine-tuning epochs.

#### **Accelerate** (`accelerate>=0.20.0`)
- **What:** Hugging Face library for easy multi-GPU and mixed-precision training.
- **Why Used:** Dependency for transformers; enables efficient model loading and device management.

---

### 4.4 Explainability Library

#### **SHAP** (`shap>=0.42.0`) — SHapley Additive exPlanations
- **What:** Game-theory based method to explain individual ML predictions.
- **Core Idea:** For each prediction, SHAP calculates how much each word contributed to the prediction.
- **Where Used:** `app.py` Explainability page — `LinearExplainer` for the baseline Logistic Regression model.

---

### 4.5 Visualization & Web Libraries

#### **Plotly** (`plotly>=5.15.0`)
- **What:** Interactive charting library with hover, zoom, and pan support.
- **Where Used:** All charts in `app.py` — bar charts, heatmaps, pie charts, histograms, grouped comparisons.

#### **Streamlit** (`streamlit>=1.28.0`)
- **What:** Framework for building data science web apps in pure Python.
- **Where Used:** `app.py` — entire dashboard (7 pages, caching, columns, metrics).

#### **Streamlit Option Menu** (`streamlit-option-menu>=0.4.0`)
- **What:** Stylized sidebar navigation with icons.
- **Where Used:** `app.py` — sidebar navigation with Bootstrap icons.

---

### 4.6 Serialization Library

#### **Joblib** (`joblib>=1.2.0`)
- **What:** Efficiently save/load Python objects (models, arrays, metrics).
- **Where Used:**
  - Training scripts — `joblib.dump()` saves metrics
  - `app.py` — `joblib.load()` loads metrics for display

---

## 5. Phase 1: Baseline Model

### Pipeline
```
Raw Text → Preprocessing → TF-IDF (50K features) → Logistic Regression → Sentiment (Pos/Neg/Neu)
```

### Training Script: `train_baseline.py`

### Step-by-Step Process

1. **Data Loading** (memory-efficient chunks)
   - Read CSV in 50,000-row chunks using `pd.read_csv(chunksize=50000)`
   - Only load `title` column to minimize memory
   - Loads the full ~1M row dataset

2. **Text Preprocessing**
   - Convert to lowercase
   - Remove URLs (`http://...`, `https://...`)
   - Remove Reddit mentions (`u/username`)
   - Remove special characters (keep only letters and spaces)
   - Collapse multiple spaces to single space

3. **Label Simulation** (since dataset has no ground truth labels)
   - Count positive keywords (moon, rocket, gains, bullish, squeeze...)
   - Count negative keywords (loss, crash, bear, puts, dump...)
   - Assign: Positive if pos > neg, Negative if neg > pos, else Neutral
   - Add 10% random noise for realistic label uncertainty (accurate models cap at ~88-90%)

4. **Train/Test Split** — 80/20 stratified

5. **TF-IDF Vectorization** — 50,000-dimensional sparse vectors, unigrams + bigrams

6. **Logistic Regression** — SAGA solver, all CPU cores

7. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report

### Saved Artifacts
- `baseline_model.joblib` — Trained Logistic Regression
- `tfidf_vectorizer.joblib` — Fitted TF-IDF
- `label_encoder.joblib` — Label encoder
- `test_data.joblib` — Test set for SHAP
- `baseline_metrics.joblib` — All metrics and timing

---

## 6. Phase 2: DistilBERT Sentiment Model

### Pipeline
```
Raw Text → Preprocessing → DistilBERT Tokenizer → DistilBERT Fine-tuning → Sentiment (Pos/Neg/Neu)
```

### Training Script: `train_distilbert_sentiment.py`

### Step-by-Step Process

1. **Data Loading** — 10,000 rows for faster training
2. **Text Preprocessing** — Same as Phase 1
3. **Label Simulation** — Same keyword heuristics
4. **Tokenization** — DistilBERT WordPiece tokenizer (128 max tokens)
5. **DataLoader** — PyTorch DataLoader with batch_size=16
6. **Fine-tuning** — 1 epoch, AdamW optimizer (lr=2e-5), gradient clipping (max_norm=1.0)
7. **Evaluation** — Accuracy on validation set, classification report

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `distilbert-base-uncased` |
| Max Length | 128 tokens |
| Batch Size | 16 |
| Epochs | 1 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Scheduler | Linear warmup |
| Output Labels | 3 (Positive, Negative, Neutral) |

### How DistilBERT Training Works

1. **Pre-trained model** loaded from Hugging Face — already knows English language structure
2. **Classification head** added on top (3-class linear layer)
3. **Fine-tuning** adjusts ALL model weights on our financial text data
4. **Forward pass:** tokenized text → transformer layers → [CLS] token → classifier → logits
5. **Loss:** Cross-entropy loss between predicted and true labels
6. **Backpropagation:** Gradients flow through entire model, updating weights

### Saved Artifacts
- `distilbert_sentiment/model.safetensors` — Fine-tuned model weights
- `distilbert_sentiment/tokenizer_config.json` — Tokenizer configuration
- `distilbert_sentiment/sentiment_metrics.joblib` — Metrics
- `distilbert_sentiment/label_encoder.joblib` — Label encoder

---

## 7. Phase 3: DistilBERT Sarcasm Detection

### Pipeline
```
Raw Text → Preprocessing → DistilBERT Tokenizer → DistilBERT Fine-tuning → Sarcasm (0 or 1)
```

### Training Script: `train_distilbert_sarcasm.py`

### Step-by-Step Process

1. **Data Loading** — 10,000 rows
2. **Preprocessing & Label Simulation** — Generates both sentiment AND sarcasm labels
3. **Sarcasm Label Strategy:**
   - Detect sarcasm markers (obviously, clearly, genius, lol, lmao...)
   - Mixed-signal detection (positive words in negative context or vice versa)
   - Short punchy text with punctuation
   - ~20% sarcasm rate in dataset
4. **Fine-tuning** — Same architecture, but with 2 output labels (sarcastic / not sarcastic)
5. **Evaluation** — Binary classification metrics

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `distilbert-base-uncased` |
| Max Length | 128 tokens |
| Batch Size | 16 |
| Epochs | 1 |
| Learning Rate | 2e-5 |
| Output Labels | 2 (Not Sarcastic, Sarcastic) |

### How Sarcasm Adjustment Works (Live Prediction)

When making live predictions in the dashboard:
1. Text is preprocessed and tokenized
2. **Sarcasm model** predicts if text is sarcastic
3. **Sentiment model** predicts base sentiment
4. If sarcastic → sentiment polarity is **flipped**:
   - Positive → Negative (sarcastic praise = actually negative)
   - Negative → Positive (sarcastic complaint = actually positive)
   - Neutral → Neutral (no change)

---

## 8. Phase 4: Explainable AI (SHAP)

### What is SHAP?
**SHAP (SHapley Additive exPlanations)** uses game theory to explain individual predictions by computing each feature's (word's) contribution to the prediction.

### Why SHAP Matters
1. **Transparency** — Understand WHY the model made a specific prediction
2. **Trust** — Verify the model uses sensible features
3. **Debugging** — Identify spurious correlations
4. **Research** — Demonstrate which financial words drive sentiment

### Implementation
```python
explainer = shap.LinearExplainer(
    model,                                    # Baseline LogReg model
    X_sample_tfidf,                          # Background data
    feature_perturbation="interventional"     # Independence assumption
)
shap_values = explainer.shap_values(X_sample_tfidf)
```

> **Note:** SHAP is applied to the baseline Logistic Regression model (interpretable linear model). DistilBERT predictions use the Live Text Prediction page instead.

### SHAP Visualizations

| Visualization | What It Shows |
|---------------|---------------|
| **Global Feature Importance** | Top 20 words ranked by mean absolute SHAP value |
| **Per-Class Importance** | Top 10 words for each class (Positive/Negative/Neutral) |
| **Summary Plot** | Distribution of SHAP values per feature |

---

## 9. Dashboard (Streamlit)

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------:|
| Backend | Streamlit (Python) | Web framework |
| Charts | Plotly | Interactive, dark-themed visualizations |
| Navigation | streamlit-option-menu | Styled sidebar menu |
| Styling | Custom CSS | Dark theme with gradient cards |
| ML Inference | PyTorch + Transformers | Live DistilBERT predictions |
| Data | Joblib | Load saved model artifacts |

### 7 Dashboard Pages

| # | Page | Content |
|---|------|---------:|
| 1 | **Home** | Project overview, methodology table, system architecture |
| 2 | **Dataset Overview** | Dataset statistics (1M rows), column info, sample data, title length distribution |
| 3 | **Baseline Model** | DistilBERT Sentiment results — accuracy, confusion matrix, config, timing, classification report |
| 4 | **Sarcasm Detection** | DistilBERT Sarcasm results — accuracy, confusion matrix, classification report |
| 5 | **Accuracy Comparison** | **Baseline (TF-IDF + LR) vs DistilBERT** — accuracy bar chart, all-metrics comparison, detailed table, side-by-side confusion matrices, classification reports, training time comparison, key findings |
| 6 | **Live Text Prediction** | Enter text → DistilBERT predicts sentiment + sarcasm → shows sarcasm-aware final prediction |
| 7 | **Explainability** | SHAP analysis on baseline model — top 20 words, per-class importance, summary plot |

### Custom CSS Design
- **Gradient metric cards** — Purple, green, pink, blue, teal gradient backgrounds
- **Section headers** — Dark navy background with blue accent border
- **Info/Success boxes** — Dark cards with colored left border
- **Dark tables** — All dataframes styled with dark backgrounds (#161b22)
- **Plotly dark theme** — Transparent paper, #161b22 plot background, light text

---

## 10. Data Pipeline & Preprocessing

### Text Preprocessing Functions

```python
def preprocess_text(text):
    text = str(text).lower()                          # 1. Lowercase
    text = re.sub(r'http\S+|www\.\S+', '', text)     # 2. Remove URLs
    text = re.sub(r'u/\w+', '', text)                 # 3. Remove Reddit mentions
    text = re.sub(r'\$([a-zA-Z]+)', r'\1', text)     # 4. Remove $ from tickers
    text = re.sub(r'[^a-z\s]', '', text)             # 5. Keep only letters + spaces
    text = re.sub(r'\s+', ' ', text).strip()          # 6. Normalize spaces
    return text
```

### DistilBertDataset (PyTorch)

```python
class DistilBertDataset(Dataset):
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
```

### Label Simulation Strategy

**Sentiment Labels:**
- Match text against positive keywords (moon, rocket, gains, bullish...)
- Match text against negative keywords (loss, crash, bear, puts...)
- Assign based on dominant polarity; default to Neutral
- Add 10% random noise for realism

**Sarcasm Labels:**
- Sarcasm marker words (obviously, clearly, genius, lol, lmao...)
- Mixed-signal detection (positive words in negative context or vice versa)
- Short punchy text with punctuation
- ~20% sarcasm rate

---

## 11. Results & Findings

### Model Comparison

| Model | Approach | Accuracy | Training Data | Training Time |
|-------|----------|:--------:|:-------------:|:-------------:|
| **Baseline** | TF-IDF + Logistic Regression | ~85.10% | ~1M rows | ~77s |
| **DistilBERT Sentiment** | Fine-tuned Transformer | 90.30% | 10K rows (1 epoch) | ~161s |
| **DistilBERT Sarcasm** | Fine-tuned Transformer (Binary) | 96.90% | 10K rows (1 epoch) | ~150s |
| **Sarcasm-Aware Pipeline** | **Sequential Ensemble** | **87.90%** | **2,000 holdout** | **82s (Eval)** |

> **Note:** Actual accuracy values depend on training runs. View the **Accuracy Comparison** page in the dashboard for exact numbers.

### Key Analysis: Why Similar Accuracy?

The DistilBERT model may show accuracy close to the TF-IDF baseline because:

1. **Only 1 epoch** — Transformers typically need 3-5 epochs to fine-tune properly
2. **10K rows vs 1M rows** — The baseline trains on 100× more data
3. **Keyword-based labels** — Since labels are generated from keyword heuristics, TF-IDF (which is literally a keyword frequency method) captures the labeling pattern perfectly. DistilBERT's contextual understanding doesn't help when labels are keyword-driven.

### How to Improve DistilBERT Performance

| Change | Current | Recommended |
|--------|---------|-------------|
| Epochs | 1 | 3–5 |
| Training rows | 10,000 | 50,000+ |
| Labels | Simulated (keyword) | Real annotations or FinBERT-generated |

---

## 12. How to Run the Project

### Prerequisites
- Python 3.9 or higher
- ~4GB RAM minimum (for DistilBERT models)
- GPU optional (falls back to CPU automatically)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Dataset (if CSV not present)
```bash
python generate_data.py
```
Creates `r_wallstreetbets_big.csv` with 1M simulated Reddit posts.

### Step 3: Train Phase 1 — Baseline Model (TF-IDF + LR)
```bash
python train_baseline.py
```
Output: Saved model artifacts in `models/` directory (~77 seconds on 1M rows)

### Step 4: Train Phase 2 — DistilBERT Sentiment Model
```bash
python train_distilbert_sentiment.py
```
Output: Saved model in `models/distilbert_sentiment/` (time varies by hardware)

### Step 5: Train Phase 3 — DistilBERT Sarcasm Model
```bash
python train_distilbert_sarcasm.py
```
Output: Saved model in `models/distilbert_sarcasm/` (time varies by hardware)

### Step 6: Launch Dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` — navigate all 7 pages to explore results.

### Quick Verification
After training all models, the dashboard should show:
- ✅ **Baseline Model** page — metrics and confusion matrix
- ✅ **Sarcasm Detection** page — sarcasm model results
- ✅ **Accuracy Comparison** page — Baseline vs DistilBERT side-by-side
- ✅ **Live Text Prediction** — Enter text and get real-time predictions

---

*📊 Financial Text Sentiment Analysis with Sarcasm Detection & Explainable AI*
*Built with Python, Scikit-learn, PyTorch, Transformers, SHAP, Plotly & Streamlit*
