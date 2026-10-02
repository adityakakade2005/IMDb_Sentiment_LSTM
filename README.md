
**Notebook:** `IMDb_Sentiment_LSTM_Reproduction(2).ipynb`

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

Real-time movie analysis by name

Given just a movie name, Cell 17b answers: what genre is it, what's its IMDb rating, and what do real reviews say about it — according to this model?

The model itself only understands review text, not titles, so this cell fetches real-world data from two sources and combines them:

Data	Source	Why
Genre, year, plot, IMDb rating	OMDb API	Free API that legally serves data straight from IMDb's own database — this really is IMDb's genre/rating, not an approximation.
Written review text, TMDB rating, vote count, poster image	TMDB API	IMDb has no public API for individual review text, and scraping IMDb's review pages directly violates their Terms of Service, so TMDB (free, public, ToS-compliant) is used for review text instead — its own rating, vote count, and poster come back in that same call at no extra cost.

Each fetched review is scored with predict_sentiment(), and the cell prints an overall verdict (Mostly Positive / Mostly Negative / Mixed) based on how many of the fetched reviews came out positive vs. negative.

Note on the standard Python IMDb library (cinemagoer/IMDbPy): current versions of this library only expose IMDb's official structured datasets (title, year, genre, rating) — the review-scraping functionality it once had has been removed. That's why this notebook uses OMDb (for that same structured IMDb data) plus TMDB (for review text) rather than that library.

One-time setup (two free API keys):

OMDb key → https://www.omdbapi.com/apikey.aspx (free tier, delivered instantly by email)
TMDB key → https://www.themoviedb.org/settings/api (free, requires an account)
In Cell 17b, set OMDB_API_KEY and TMDB_API_KEY to your own keys, and movie_name to whatever you want to analyze.

Limitations:

Not every title has written reviews on TMDB — genre/rating will still print even if no reviews are found.
OMDb can occasionally fail to resolve very ambiguous, misspelled, or extremely obscure titles — try the exact/full title if this happens.
TMDB reviews are written by TMDB's own userbase, not IMDb's, so wording/style may differ from the training dataset.
Interactive UI

Cell 20 is an ipywidgets-based front end for everything above: a text box for the movie name, a slider for how many reviews to pull, and an Analyze button. Clicking it runs analyze_movie() (Cell 17b) and renders a styled report directly in the notebook — poster image, genre, IMDb rating, TMDB rating, plot, a color-coded sentiment badge per review, and an overall verdict banner — without editing any code per movie.

To use it:

Run every cell above it in order (model must be trained, OMDB_API_KEY/TMDB_API_KEY must be set in Cell 17b).
Run Cell 20. A small panel appears with the input box, slider, and button.
Type a movie name, adjust "Max reviews" if desired, click Analyze.

Environment notes:

Works out of the box in Jupyter Notebook / JupyterLab.
In Google Colab, widgets sometimes need to be explicitly enabled first — if the UI doesn't render, add this in a cell above Cell 20 and rerun:
python
  from google.colab import output
  output.enable_custom_widget_manager()
This cell is purely a UI layer — it reuses the same analyze_movie() pipeline from Cell 17b, so all of that cell's limitations (API keys required, some titles may have no reviews) apply here too.

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
