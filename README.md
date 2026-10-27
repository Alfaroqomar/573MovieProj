# MovieMatch

Classical movie recommendations on **MovieLens 32M**, enhanced by LLM reranking and grounded explanations.

```text
User ratings
      ↓
Classical recommendation models: Popularity / Item-KNN / ALS
      ↓
Top 20–100 candidate movies (50 by default, from ALS)
      ↓
LLM assistance: query interpretation · candidate reranking · grounded explanations
      ↓
Validated final Top-N recommendations
```

Everything shown in the app is computed from the real dataset and the models trained on it. There is no mock
data and no sample-data fallback: if the dataset or a model is missing, the app says so and tells you which
command to run.

---

## Requirements

| | |
|---|---|
| Python | 3.10 or newer |
| Node.js | 20 or newer |
| RAM | 8 GB minimum, 16 GB recommended (preprocessing holds all 32M ratings in memory) |
| Disk | about 1 GB for the extracted dataset plus about 1 GB for artifacts |

## Setup

### 1. Download MovieLens 32M

Download and extract the official ZIP (about 239 MB):

- Dataset page: https://grouplens.org/datasets/movielens/32m/
- README and license: https://files.grouplens.org/datasets/movielens/ml-32m-README.html
- ZIP: https://files.grouplens.org/datasets/movielens/ml-32m.zip

```bash
curl -LO https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip            # creates ml-32m/ with ratings.csv, movies.csv, tags.csv, links.csv
```

### 2. Configure

```bash
cp .env.example .env
```

Set `MOVIELENS_DATA_PATH` to the **absolute** path of the extracted `ml-32m` folder. All other settings are
optional. `.env` is git-ignored; never commit it.

### 3. Install

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..
```

(or `make install`)

## Commands

Run from the repository root. Each step is separate, so models are never retrained when the app starts.

| Step | Command | Make shortcut |
|---|---|---|
| Check readiness | `cd backend && python -m moviematch.cli status` | `make status` |
| **Preprocess dataset** | `cd backend && python -m moviematch.cli preprocess` | `make preprocess` |
| **Train models** | `cd backend && python -m moviematch.cli train` | `make train` |
| Evaluate (optional) | `cd backend && python -m moviematch.cli evaluate` | `make evaluate` |
| **Start backend** | `cd backend && python -m uvicorn moviematch.api.main:app --port 8000` | `make backend` |
| **Start frontend** | `cd frontend && npm run dev` | `make frontend` |
| Backend tests | `cd backend && python -m pytest` | `make test` |

Then open http://localhost:5173. The Vite dev server proxies `/api` to the backend at `http://127.0.0.1:8000`
(override with `MOVIEMATCH_BACKEND_URL` when starting the frontend).

Useful options:

- `train --only als` (or `popularity`, `knn`) retrains a single model.
- `evaluate --split validation` evaluates on the validation split (for tuning); `--users 2000` changes the
  sample size; `--k 20` changes the cutoff.
- Restart the backend after preprocessing or training so it loads the new artifacts.

Rough timings on a laptop: preprocessing a few minutes, training a few minutes (ALS dominates), evaluation of
10,000 users about a minute.

## Using an LLM (optional)

Reranking works with any server that implements the OpenAI-compatible `POST /chat/completions` API. Configure
it in `.env`; nothing about the provider is hard-coded, and the key never reaches the browser.

```env
LLM_ENABLED=true
LLM_BASE_URL=http://localhost:11434/v1      # example: a local Ollama server
LLM_API_KEY=                                # set if your provider requires one
LLM_MODEL=                                  # the model name your provider expects
LLM_TIMEOUT_SECONDS=60
LLM_JSON_MODE=true                          # set false if the provider rejects response_format
```

For ASU Voyager or a hosted open-weight model, use that service's OpenAI-compatible base URL, key, and model
name. If the LLM is not configured, the **ALS + LLM Reranking** option is disabled, the UI shows
“LLM reranking is not configured.”, and Popularity, Item-KNN, and ALS keep working.

## What each part does

### Preprocessing (`backend/moviematch/data/preprocess.py`)

1. Verifies `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv` exist and have the expected columns.
2. Drops invalid rows: missing values, ratings outside 0.5–5.0 or not in 0.5 steps, negative timestamps,
   ratings for movies not in `movies.csv`, empty tags. Counts are recorded in `manifest.json`.
3. Deduplicates: repeated (user, movie) ratings keep the most recent; tags are normalized (trimmed,
   lower-cased) and deduplicated per user.
4. Maps user and movie IDs to contiguous internal indices (every movie in `movies.csv` gets one).
5. Sorts each user's ratings by timestamp (ties broken by movie ID) and splits them chronologically:
   earliest 80% → train, next 10% → validation, latest 10% → test (at least one training rating per user).
6. Writes Parquet tables (movies, ratings with split labels, users, user top movies, aggregated tags),
   sparse matrices (`train_matrix.npz`, `full_matrix.npz`), `statistics.json`, and `manifest.json` with the
   dataset version, source file sizes, cleaning report, and configuration.

Titles are displayed as `The Matrix` / `1999` rather than `Matrix, The (1999)`; the raw title is kept too.

### Models (`backend/moviematch/models/`)

All models are trained on the **training split only**. Validation and test ratings never reach training.

**Popularity.** Bayesian weighted rating on training ratings:

```text
WR(i) = v_i / (v_i + m) · R_i + m / (v_i + m) · C
```

`v_i` = training ratings for movie *i*, `R_i` = its mean training rating, `C` = mean of all training ratings,
`m` = the 90th percentile of `v_i` over rated movies (`POPULARITY_MIN_VOTES_QUANTILE`). Movies with
`v_i ≥ m` are ranked first by WR; the remaining rated movies follow so very active users still get results.
Movies the user has rated are skipped.

**Item-KNN.** Cosine similarity between movies' training rating vectors, computed with the sparse,
multi-threaded all-pairs KNN in the `implicit` library. Only the top `KNN_NEIGHBORS` (50) neighbours per movie
are stored, so no dense 87,585 × 87,585 matrix is built. Movies with fewer than `KNN_MIN_ITEM_RATINGS` (20)
training ratings get no neighbours. The matrix is symmetrized (max(S, Sᵀ)) with a zero diagonal. A user's score
for movie *j* is `Σ r_ui · S[i, j]` over the movies *i* they rated ≥ 4.0 (all rated movies if none). Evidence is
the highly rated history movies with the largest `S[i, j]`. The KNN score is a raw weighted sum, not a
probability.

**ALS.** `implicit`'s conjugate-gradient ALS (64 factors, λ = 0.05, 15 iterations by default). Ratings are
treated as **implicit feedback**: each rated movie is a positive interaction with confidence
`c = 1 + ALS_ALPHA · rating` (α = 2.0), so higher ratings weigh more; unrated movies are preference 0 with
confidence 1. When serving, the user's vector is re-solved from their full rating history with the trained item
vectors held fixed (no retraining). The top 50 unseen movies form the candidate pool. The displayed **ALS
score** is min-max normalized across that pool (1.0 = strongest candidate, 0.0 = weakest); the raw dot product
is also returned. It is a relative score, not a probability.

### LLM reranking (`backend/moviematch/llm/`)

1. ALS produces the top-50 unseen candidates.
2. Item-KNN attaches up to two similarity links per candidate to the user's highly rated movies.
3. The backend sends compact JSON: the request text, the user's highly rated movies and favourite genres, and
   per candidate its ID, title, year, MovieLens genres, top tags, ALS score and rank, and KNN evidence.
4. The LLM must return strict JSON with `interpretedRequest` and `rankedMovies`.
5. The backend validates the response:
   strict JSON parse (one enclosing ```` ```json ```` fence is tolerated; NaN/Infinity rejected) → Pydantic
   schema with no extra fields → every `movieId` in the candidate list → no duplicates → exactly N results →
   ranks unique and sequential from 1 → scores finite, clamped to [0, 1] → explanations non-empty and at most
   300 characters → interpreted genres restricted to MovieLens genre names. Failures are logged without
   prompts or secrets.
6. On a validation failure the backend makes **one** correction call listing the errors. If that fails too,
   or the LLM times out or errors, the result is the original ALS order marked
   **“ALS fallback because LLM reranking failed.”** with the reason. No LLM output is ever fabricated.

The LLM never generates candidates and cannot introduce a movie outside the ALS pool.

### Evaluation (`backend/moviematch/models/evaluation.py`)

On a fixed-seed sample of users with at least one held-out rating ≥ 4.0: model input = training ratings,
previously seen movies (train + validation for the test split) excluded, relevant = held-out ratings ≥ 4.0.
Reports Precision@10, Recall@10, NDCG@10, HitRate@10, MRR@10 and mean latency to
`artifacts/evaluation/results.json`, which the Compare page displays. Until it is run, the page shows
“Evaluation not run yet.” ALS + LLM is not evaluated offline (it would need one LLM call per user).

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | |
| GET | `/api/config/status` | readiness state, available modes, default mode, LLM status (never the key or URL) |
| GET | `/api/dataset/status` | raw files found, processed, manifest |
| GET | `/api/dataset/statistics` | counts from the processed dataset, most-rated movies |
| GET | `/api/users/search?query=&page=&pageSize=` | prefix search on user IDs, paginated server-side |
| GET | `/api/users/{userId}/summary` | rating count, favourite genres, highly rated movies |
| GET | `/api/movies/{movieId}` | title, year, genres, tags, rating stats, IMDb/TMDB IDs from links.csv |
| POST | `/api/recommendations/popularity` | `{"userId": 42, "topN": 10}` |
| POST | `/api/recommendations/knn` | `{"userId": 42, "topN": 10}` |
| POST | `/api/recommendations/als` | `{"userId": 42, "topN": 10, "candidateCount": 50}` |
| POST | `/api/recommendations/als-llm` | `{"userId": 42, "topN": 10, "candidateCount": 50, "userRequest": "..."}` |
| GET | `/api/evaluation/results` | `{"available": false, "message": "Evaluation not run yet."}` until evaluated |

Add `?stream=true` to any recommendation endpoint to receive newline-delimited JSON progress events
(`loading_history`, `generating_als`, `finding_similar`, `reranking_llm`, `validating`) followed by the
result. The UI uses this to show real progress. Errors are `{"error": {"code", "message"}}` with no stack
traces. Interactive docs: http://127.0.0.1:8000/docs.

### Readiness states

| State | Meaning | What to do |
|---|---|---|
| `dataset_unavailable` | `MOVIELENS_DATA_PATH` unset, wrong, or files missing | Set the path, run preprocess |
| `not_processed` | Raw files found, no processed artifacts | Run preprocess, then train |
| `models_not_trained` | Processed, no models | Run train, restart backend |
| `partial` | Some models missing or failed to load | Run train, restart backend |
| `ready` | All three models loaded | LLM status is reported separately |

## Posters

MovieLens has no poster images. Cards show a generated placeholder with the title and genres. To use a poster
source you are permitted to use, set `POSTER_URL_TEMPLATE` (placeholders `{movieId}`, `{imdbId}`,
`{tmdbId}`). Posters that fail to load fall back to the placeholder and never block recommendations.

## Project structure

```text
backend/
  moviematch/
    cli.py              status / preprocess / train / evaluate
    config.py           environment-variable settings
    service.py          readiness checks, loaded artifacts, recommendation orchestration
    data/               raw-file checks, preprocessing, artifact paths
    models/             popularity, item_knn, als, training, evaluation
    llm/                OpenAI-compatible client, response schema + validation, reranker
    api/                FastAPI app and Pydantic API schemas
  tests/                pytest suite (uses a tiny synthetic MovieLens-format fixture, test-only)
frontend/
  src/
    api/                typed client (fetch + NDJSON progress stream)
    components/         layout, cards, details dialog, user picker, pipeline, progress
    pages/              Home, Recommendations, Compare models, About
.env.example            all settings, no secrets
Makefile                shortcuts for the commands above
```

## Testing

`cd backend && python -m pytest` runs 35 tests covering: the chronological split (no train/validation/test
time overlap), row cleaning and deduplication, exclusion of already-rated movies in all three models, ALS score
normalization, evaluation metrics, every LLM validation rule, the one-retry-then-fallback behaviour, timeouts,
that the LLM receives exactly the ALS top-50, that secrets are not sent or echoed, readiness states, and the
streaming endpoints.

The tests generate a small, clearly synthetic dataset in the MovieLens file format inside a temporary
directory. It exists only for the tests; the application never uses or falls back to it.

## Limitations

- MovieLens has no plots, cast, directors, or posters; explanations can only cite genres, tags, ratings, and
  similarity evidence.
- The LLM can only reorder 50 candidates; it cannot recover a movie ALS did not propose.
- Validation checks the LLM's structure and IDs, not every sentence of its explanations.
- Tags are community-applied and sparse; many movies have none. Tags are used as content metadata and are not
  split by time.
- Users with a single rating have nothing in validation or test.
- NeuMF and other neural models are future work.

## Data license

MovieLens 32M is provided by GroupLens Research under its own terms (see the dataset README). The dataset is not
included in this repository.
