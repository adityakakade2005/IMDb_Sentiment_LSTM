# IMDb Sentiment Analysis — LSTM Reproduction

A self-contained Jupyter notebook that reproduces **Varadharajan, Smith, Kalla, Samaah & Mandala (2025)**, *"Deep Learning-Based Sentiment Analysis: Enhancing IMDb Review Classification with LSTM Models"* — three classical TF-IDF baselines plus a stacked LSTM, benchmarked against the paper's reported numbers.

**Notebook:** `IMDb_Sentiment_LSTM_Reproduction.ipynb`

---

## What it does

1. Loads the 50,000-review IMDb dataset (auto-downloads if not present locally).
2. Cleans and lemmatizes text (HTML/URL stripping, stopword removal, WordNet lemmatization).
3. Runs EDA on review length distributions and class balance.
4. Performs a stratified 80/20 train/test split.
5. Trains three classical baselines on TF-IDF features:
   - Gaussian Naive Bayes
   - Multinomial Naive Bayes
   - Decision Tree
6. Tokenizes/pads text (`MAX_LENGTH = 80`) and trains a stacked LSTM:
   `Embedding(50) → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(2, softmax)`
7. Evaluates every model (accuracy, precision, recall, F1, confusion matrix) and compares results against the paper's reported figures.
8. Wraps inference in a reusable `predict_sentiment(text)` function and tests it on sample reviews.
9. Serializes the trained model (`imdb_lstm_sentiment.keras`) and tokenizer (`tokenizer.pkl`) for later reuse.
10. Closes with a parameter-provenance table (paper-specified vs. implementation-choice hyperparameters) and a short discussion for viva/defense purposes.

---

## Requirements

- Python 3.9–3.12
- Packages:
  ```
  numpy
  pandas
  matplotlib
  seaborn
  nltk
  scikit-learn
  tensorflow          # or tensorflow-cpu
  ```

Install with:

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn tensorflow
```

(NLTK corpora — `stopwords`, `wordnet`, `omw-1.4` — are downloaded automatically by Cell 1; this requires an internet connection the first time.)

---

## How to run

1. Open the notebook in Jupyter, JupyterLab, VS Code, or Google Colab.
2. **Run all cells in order, top to bottom.** Later cells depend on variables created earlier (`df`, `tokenizer`, `model`, etc.), so skipping cells or running out of order will raise `NameError`s.
3. On the first run, Cell 2 downloads `IMDB Dataset.csv` (~65 MB) into the working directory; subsequent runs reuse the local copy.
4. Training the LSTM (Cell 13) is the slowest step — a few minutes on CPU, faster with a GPU.

---

## Outputs produced when run

| File | Produced by | Purpose |
|---|---|---|
| `IMDB Dataset.csv` | Cell 2 | Raw dataset, cached locally |
| `imdb_lstm_sentiment.keras` | Cell 18 | Trained LSTM model |
| `tokenizer.pkl` | Cell 18 | Fitted Keras tokenizer for inference |

---

## Notes on reproduction fidelity

The notebook distinguishes two categories of hyperparameters throughout (see Cell 19's provenance table):

- **Explicitly specified by the paper** — sequence length (80), embedding dimension (50), LSTM layer sizes (64/32), dropout (0.2), output layer, and training epochs (5).
- **Implementation choices** — not stated in the paper, so a reasonable, reproducible default is used and labeled as such: random seed (42), vocabulary size (10,000), train/test split ratio, optimizer (Adam, lr=0.001), loss function, and batch size (64).

Exact accuracy numbers will vary slightly run-to-run and from the paper's reported figures due to dataset version differences, randomness in weight initialization, and the unreported hyperparameters above — the notebook prints the delta against each paper-reported metric so this is transparent rather than silently glossed over.

---

## Troubleshooting

- **NLTK download fails / hangs** — requires internet access on first run; rerun Cell 1 after connecting.
- **MemoryError on the Gaussian NB cell** — the dense TF-IDF array is a few hundred MB; reduce `max_features` in Cell 6 if needed.
- **Very slow LSTM training** — expected on CPU-only machines; using `tensorflow-cpu` still works, just budget more time, or use a GPU-enabled runtime (e.g., Colab).
