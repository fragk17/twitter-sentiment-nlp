# Twitter Sentiment Classification: from TF-IDF to BERT

Binary sentiment classification (positive / negative) of English tweets. I built and compared **three increasingly powerful approaches** over a semester, each one tuned through systematic experiments and evaluated with accuracy, precision, recall, F1, ROC curves, learning curves and confusion matrices.

> *Artificial Intelligence II: Deep Learning for NLP* (Dept. of Informatics & Telecommunications, University of Athens, Spring 2025). The data came from a course Kaggle competition.

## Results

| # | Approach | Representation | Validation accuracy |
|---|---|---|---|
| 1 | Logistic Regression (liblinear, C=1.2, L2) | TF-IDF n-grams | 0.8025 |
| 2 | Feed-forward neural network (PyTorch) | averaged GloVe Twitter embeddings (200d) | 0.7890 (test: 0.7882) |
| 3 | **Fine-tuned BERT** (`bert-base-uncased`) | contextual embeddings | **0.8543** (F1 0.853) |
| 3b | Fine-tuned DistilBERT | contextual embeddings | 0.8467 (F1 0.846) |

**Takeaway:** averaging word vectors throws away word order, so the DNN could not beat a well-tuned TF-IDF baseline that captures n-grams. Fine-tuned transformers outperformed both by about 5 points. DistilBERT came within 1 point of BERT while being much smaller and faster.

## Neural network pipeline (this repo's code)

[`src/final_dnn.py`](src/final_dnn.py)

1. **Twitter-specific preprocessing.** Lowercasing, `@mentions` → `MENTION`, URLs → `URL`, squashing repeated characters (`soooo` → `soo`), punctuation and whitespace cleanup.
2. **Vectorization.** Each tweet becomes the average of its **GloVe Twitter 200d** vectors (loaded with `gensim`).
3. **Model.** A 3-hidden-layer MLP (256 → 128 → 64) with **BatchNorm**, **LeakyReLU** and **Dropout**.
4. **Training.** `BCEWithLogitsLoss` with a positive-class weight for class imbalance, Adam with weight decay, **ReduceLROnPlateau** on validation F1, gradient clipping, **early stopping** (patience 3) and fixed seeds for reproducibility.

The experiments behind it, with each change and its effect:

| Trial | Preprocessing | Embeddings | Model | Accuracy |
|---|---|---|---|---|
| Baseline ([`src/baseline_dnn.py`](src/baseline_dnn.py)) | none | GloVe 6B 50d | 2 layers, Adam | 0.699 |
| 2 | stop-words + lemmatization | GloVe Twitter 100d | 2 layers, AdamW | 0.743 |
| Final | Twitter-specific cleaning | GloVe Twitter 200d | 3 layers + BN, Adam, LR scheduler, early stopping | **0.789** |

## BERT / DistilBERT

The transformer models were fine-tuned with Hugging Face Transformers on Kaggle GPUs: AdamW (lr 2e-5), a linear learning-rate scheduler (with warm-up for DistilBERT), short max sequence lengths, and model selection by validation F1.
Notebooks: [BERT](https://www.kaggle.com/code/fragkakis/sdi2100206-bert-hw3) · [DistilBERT](https://www.kaggle.com/code/fragkakis/sdi2100206-distilbert-hw3)

## Reports

The full write-ups, with ROC curves, learning curves, confusion matrices and error analysis:

- [`reports/dnn_glove_report.pdf`](reports/dnn_glove_report.pdf): neural network with GloVe embeddings
- [`reports/bert_distilbert_report.pdf`](reports/bert_distilbert_report.pdf): BERT and DistilBERT fine-tuning

## Running

The scripts were written as Kaggle notebooks and read data from `/kaggle/input/...`. To run them elsewhere, update the paths at the top of the script.

```bash
pip install -r requirements.txt
python src/final_dnn.py
```

**Tech:** Python · PyTorch · scikit-learn · gensim · NLTK · pandas · NumPy · Hugging Face Transformers

## Author

**Giorgos Fragkakis** · [GitHub](https://github.com/fragk17) · [Kaggle](https://www.kaggle.com/fragkakis)
