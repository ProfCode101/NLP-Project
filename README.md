# arXiv CS Abstract Classification

Multi-class text classification of arXiv computer science paper abstracts into their research sub-field, using a 10-model pipeline spanning classical ML, recurrent neural networks, and a fine-tuned transformer.

## Problem

Given only the abstract text of a paper, predict which of 6 CS sub-fields it belongs to:

- Machine Learning
- Computer Vision and Pattern Recognition
- Computation and Language (NLP)
- Artificial Intelligence
- Machine Learning (Statistics)
- Neural and Evolutionary Computing

These classes are semantically close and share heavy vocabulary overlap (nearly every abstract uses words like "model", "propose", "learning"), which is what makes this a non-trivial classification problem rather than a simple topic-separation task.

## Models

| Family | Models | Representation |
|---|---|---|
| Classical ML | Random Forest, Logistic Regression, Naive Bayes | TF-IDF |
| Recurrent NN | SimpleRNN, GRU, LSTM, BiSimpleRNN, BiGRU, BiLSTM | Self-trained Word2Vec |
| Transformer | BERT-base (fine-tuned) | BERT's own subword tokenizer |

Every model is tuned across **3 hyperparameter configurations**, selected by validation macro-F1 (not accuracy, given the class imbalance — see below), then evaluated once on a held-out test set.

## Pipeline / Notebooks

The project is split into 5 notebooks, run in order, **each in its own kernel session**:

```
01_preprocessing_and_split.ipynb   Load, clean, dedupe, handle class imbalance, split
02_classical_ml.ipynb              TF-IDF + Random Forest / Logistic Regression / Naive Bayes
03_rnn_family.ipynb                Word2Vec + SimpleRNN / GRU / LSTM (+ bidirectional)
04_bert.ipynb                      BERT-base fine-tuning
05_final_comparison.ipynb          Combined results table & chart across all 10 models
```

**Why separate notebooks/kernels instead of one big notebook:** TensorFlow never releases GPU memory once it claims it. Running BERT (PyTorch) in the same process afterward can cause the two frameworks to fight over the same VRAM, which crashes the kernel with no clean Python error. Splitting by framework avoids this entirely, and also means a crash in one notebook doesn't cost you the others.

All state is handed off between notebooks through files on disk, not kernel memory — see below.

## What gets saved (and why)

| File / folder | Written by | Purpose |
|---|---|---|
| `arxiv_cleaned_full.csv`, `arxiv_cleaned_capped.csv` | Notebook 1 | Cleaned dataset, before and after class-imbalance capping |
| `train_df.csv`, `val_df.csv`, `test_df.csv` | Notebook 1 | The stratified split, reused by every later notebook |
| `label_encoder.joblib`, `class_weight_dict.joblib` | Notebook 1 | Label mapping and class weights, so every notebook encodes labels identically |
| `tuning_log.csv` | Notebooks 2/3/4 | Every hyperparameter run's validation metrics, across all models |
| `final_results.csv` | Notebooks 2/3/4 | Each model's best-config test-set metrics |
| `saved_models/classical/*.joblib` | Notebook 2 | Trained RF/LR/NB models |
| `saved_models/rnn/*.keras` | Notebook 3 | Trained RNN-family models |
| `saved_models/bert/<hash>/` | Notebook 4 | Fine-tuned BERT checkpoints (HuggingFace format) |

**Every trained model is saved immediately after training, and reloaded instead of retrained if you re-run a cell with the same config.** This matters most for BERT, where each config takes 15–20+ minutes — a kernel crash or restart no longer costs you that training time.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### GPU (RTX 3060 / WSL2 Ubuntu)

TensorFlow's GPU support requires WSL2 on Windows (native Windows GPU support was dropped after TF 2.10). Install with the CUDA extra:

```bash
pip install "tensorflow[and-cuda]"
```

If `tf.config.list_physical_devices('GPU')` still returns `[]` after this, it's almost always because the CUDA/cuDNN libraries pip installed aren't on the system's library search path. Fix:

```bash
export LD_LIBRARY_PATH=$(find $(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia -maxdepth 2 -type d -name lib | tr '\n' ':')$LD_LIBRARY_PATH
```

Make it permanent by appending that line to `.venv/bin/activate`, or system-wide via `/etc/ld.so.conf.d/` + `ldconfig` (recommended — works for every tool, not just one venv's terminal sessions).

Verify:
```bash
nvidia-smi                                                              # confirms the driver/GPU itself
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Key design decisions

- **Preprocessing:** three cleaning strategies were compared (stopwords removed + lemmatized, stopwords kept + lemmatized, stopwords removed + stemmed) on vocabulary size and average document length before picking stopword-removal + lemmatization as the final approach — see Notebook 1 for the full comparison and justification.
- **Class imbalance:** the 6 chosen classes have a ~12:1 size ratio. Majority classes are capped via stratified subsampling (`CAP = 12000` per class), and residual imbalance is handled with class weights during training. Macro-F1, not accuracy, is used to select the best hyperparameter config and to compare final models, since accuracy alone would reward majority-class bias.
- **Text representation:** TF-IDF for classical ML, a self-trained Word2Vec for the RNN family (satisfies the "≥2 of TF-IDF/Word2Vec/GloVe" requirement — GloVe was not used, noted as a possible extension).
- **Hyperparameter tuning:** manual grid search, 3 configurations per model, selected by validation macro-F1. Not an automated search (GridSearchCV/Optuna) — a deliberate tradeoff given the total training cost across 10 model families, especially BERT.

## Known limitations

- Only 6 of arXiv's 15 CS categories are used (the largest ones, for adequate per-class data).
- Majority classes are subsampled, not trained on their full size — a compute/imbalance tradeoff.
- GloVe is not used as a third text representation.
- Digits are stripped during cleaning, losing some numeric tokens (e.g. "3D") that could be mildly informative for Computer Vision abstracts specifically.

## Results

See `05_final_comparison.ipynb` for the full comparison table, per-model confusion matrices (in Notebooks 2–4), and the best/worst-performing model on the test set.
