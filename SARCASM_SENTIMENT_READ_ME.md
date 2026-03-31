# 📊 FinSAP-Pipeline: Sarcasm-Aware Financial Sentiment Analysis

**DistilBERT + Super Hybrid Stacking Ensemble + Explainable AI (SHAP)**

This project contributes a state-of-the-art **Sarcasm-Aware Sentiment Analysis** pipeline for financial text (Reddit r/wallstreetbets), featuring advanced intent detection and explainable AI.

---

## 🚀 Project Highlights

- **🎯 High Accuracy**: Achieved **87.10%** Accuracy with the **Super Hybrid Stacking Ensemble**.
- **🏆 Sarcasm-Aware Pipeline**: **87.90% Accuracy** (Verified on 200,000 rows) using sequential logic and polarity flipping.
- **🎭 Sarcasm Detector**: Integrated intent detection reaching **96.40% accuracy** to correct for sarcastic praise/complaints.
- **🔍 Explainable AI**: Visual word-level contribution analysis using **SHAP** (Shapley Additive Explanations).
- **🎨 Premium Dashboard**: A high-end Streamlit interface with 8 interactive pages, dark-mode visuals, and real-time inference.

---

## 🏗️ Architecture Overview

The system uses a multi-phase approach to handle the nuances of financial social media text:

1.  **Baseline Phase**: TF-IDF + Logistic Regression (1M rows).
2.  **Transformer Phase**: Fine-tuned DistilBERT for deep semantic understanding.
3.  **Ensemble Phase**: Super Hybrid Stacking (Meta-Judge combining TF-IDF and Transformer Experts).
4.  **Logic Phase**: Sarcasm Truth Filter for final polarity adjustment.

---

## 📑 Detailed Documentation

For a deep dive into the code structure, libraries, training pipelines, and theoretical background, please see:
👉 **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**

---

## 🛠️ Quick Start

### 1. Requirements
Ensure you have Python 3.9+ and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard
Experience the full suite of models and visualizations:
```bash
streamlit run app.py
```

---

## 📁 File Structure

- `app.py`: The main Streamlit dashboard application.
- `train_stacking_sentiment.py`: Trains the top-performing Super Hybrid model.
- `train_distilbert_sentiment.py`: Fine-tunes the base Transformer model.
- `train_distilbert_sarcasm.py`: Trains the intent detection filter.
- `utils.py`: Core preprocessing and dataset utilities.

---

*Built with Python, PyTorch, Transformers, Scikit-learn, CatBoost, LightGBM, SHAP, and Streamlit.*
