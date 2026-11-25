

# MovieMatch
**CODE**: frontend & backend

**DATA**: https://grouplens.org/datasets/movielens/32m/

**EVALUATIONS**: backend/tests/

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
| Train NeuMF (optional, needs PyTorch) | `cd backend && python -m moviematch.cli train --only neumf` | `make neumf` |
| Evaluate (optional) | `cd backend && python -m moviematch.cli evaluate` | `make evaluate` |
| Export Stage 1 shortlists for the LLM stage | `cd backend && python -m moviematch.cli candidates` | `make candidates` |
| **Start backend** | `cd backend && python -m uvicorn moviematch.api.main:app --port 8000` | `make backend` |
| **Start frontend** | `cd frontend && npm run dev` | `make frontend` |
| Backend tests | `cd backend && python -m pytest` | `make test` |

Then open http://localhost:5173. The Vite dev server proxies `/api` to the backend at `http://127.0.0.1:8000`
(override with `MOVIEMATCH_BACKEND_URL` when starting the frontend).

Useful options:

- `train --only als` (or `popularity`, `knn`, `als_explicit`, `neumf`) retrains a single model.
- `train --with-validation` fits on train + validation; use it for the final test-set numbers after tuning.
- `evaluate --split validation` evaluates on the validation period (for tuning); `--users 2000` changes the
  sample size; `--k 20` changes the cutoff.
- `candidates --users 500` sets how many test users get a 50-movie shortlist file.
- Restart the backend after preprocessing or training so it loads the new artifacts.

Measured on MovieLens 32M (2-core cloud machine; a laptop is usually faster): preprocessing 50 s (3.8 GB
peak memory), popularity + Item-KNN + implicit ALS about 1.5 minutes, explicit ALS about 5 minutes, evaluation
about 30 ms per user per all models.

**NeuMF** needs PyTorch (`pip install torch`) and is the slowest model. It uses an Apple Silicon or NVIDIA GPU
when present. For a quick run, train on a sample: `NEUMF_MAX_TRAIN_RATINGS=5000000 NEUMF_EPOCHS=5 python -m
moviematch.cli train --only neumf`. On 1 million training ratings it reached validation NDCG@10 0.08 in one
round (110 s on 2 CPU cores).

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
5. Splits by **date**, as in the proposal (`SPLIT_MODE=date`, the default): ratings before 2020-01-01 →
   train, 2020-01-01 to 2021-12-31 → validation, 2022-01-01 onward → test (`SPLIT_VALIDATION_START`,
   `SPLIT_TEST_START`). `SPLIT_MODE=per_user` keeps the earlier per-user 80/10/10 split.
6. Marks **cold-start users**: 1 to `COLD_START_MAX_RATINGS` (10) training ratings, as the proposal defines.
7. Writes Parquet tables (movies, ratings with split labels, users, top-rated and 5 most recent movies per
   user, aggregated tags), sparse matrices, `statistics.json` (including split details), and `manifest.json`.

On MovieLens 32M the date split gives 27.4M train / 2.9M validation / 1.7M test ratings (85 / 9 / 5%, not the
80 / 10 / 10 the proposal estimated). 21,529 users have no ratings before 2020 and only 165 users are
cold-start by the proposal's definition, because every MovieLens user has at least 20 ratings in total.

Titles are displayed as `The Matrix` / `1999` rather than `Matrix, The (1999)`; the raw title is kept too.

### Models (`backend/moviematch/models/`)

By default every model is fitted on the **training period only** (`train --with-validation` adds the
validation period for final test runs). Test ratings never reach any model.

**Popularity (`popularity.py`).** Two rules. *Most-rated* (the proposal's baseline, mode `popularity`): movies
by number of training ratings. *Weighted* (mode `popularity_weighted`): the Bayesian weighted rating

```text
WR(i) = v_i / (v_i + m) · R_i + m / (v_i + m) · C
```

`v_i` = training ratings for movie *i*, `R_i` = its mean, `C` = mean of all training ratings, `m` = the 90th
percentile of `v_i` (`POPULARITY_MIN_VOTES_QUANTILE`).

**Item-KNN (`item_knn.py`).** Cosine similarity between movies' training rating vectors (sparse all-pairs KNN
from `implicit`), keeping the top `KNN_NEIGHBORS` (50) per movie, so no dense 87,585 × 87,585 matrix is built.
Two scorings (`KNN_SCORING`):

- `sum` (default): `Σ r_ui · S[i, j]` over the movies *i* the user rated ≥ 4.0.
- `average` (the proposal's Eq. 2): `Σ S[i, j] · r_ui / Σ |S[i, j]|` over the user's rated movies among *j*'s
  neighbours, needing at least `KNN_MIN_SUPPORT` (2) of them; ties broken by total similarity.

The default was chosen on the validation period: on MovieLens 32M, `sum` reached NDCG@10 0.124 and
`average` 0.022, because averages favour obscure movies with one or two strong neighbours. The evaluation
always reports both.

**ALS, implicit (`als.py`).** `implicit`'s conjugate-gradient ALS (64 factors, λ = 0.05, 15 iterations). Each
rated movie is a positive interaction with confidence `c = 1 + ALS_ALPHA · rating`.

**ALS, explicit (`als_explicit.py`, mode `als_explicit`).** The proposal's Eq. 1: squared error on observed
ratings, with the per-user / per-movie regularisation Spark MLlib uses (ALS-WR), solved in closed form with
NumPy (no Java or Spark needed). Movies are ranked by predicted star rating. On validation it scored NDCG@10
0.009 against 0.120 for implicit ALS, the known weakness of rating prediction for top-N lists, so Stage 1 uses
implicit ALS.

For both ALS models a user's vector is re-solved from their ratings with the movie vectors fixed (no
retraining). Displayed ALS scores are min-max normalized across the user's candidates: relative, not
probabilities.

**Stage 1 shortlist (`stage1.py`, mode `stage1`).** The proposal's candidate merging: ALS top 30 ∪ Item-KNN top
30, each model's scores min-max normalized over its own top 50, averaged when both picked a movie, topped up
from ALS ranks 31+ to exactly 50 (`STAGE1_PER_MODEL`, `STAGE1_POOL_SIZE`, `STAGE1_TOP_UP`). On validation the
shortlist holds more of the user's later liked movies (Recall@50 0.132) than ALS alone (0.127).

**NeuMF (`neumf.py`, mode `neumf`).** He et al. (2017): a GMF branch (32-d) and an MLP branch (64 → 32 → 16),
trained with binary cross-entropy on training ratings plus 4 sampled negatives each, Adam (lr 0.001), early
stopping on validation NDCG@10. It is an **independent baseline**: it produces its own top-10 lists for the
same metrics and does not add candidates to Stage 1 or the LLM stage. Users with no training ratings have no
NeuMF vector; for them the app shows the most-rated list, clearly labelled.

**Shortlist export for the LLM stage (`candidates.py`).** `python -m moviematch.cli candidates` writes
`artifacts/candidates/test_candidates.parquet` (50 rows per user: movie, merged/ALS/KNN scores and ranks,
source, KNN evidence, cold-start flag) and `test_users.jsonl` (history size, favourite genres, top-rated and
5 most recent movies, and the held-out liked movie IDs for scoring).

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

Every model is scored on the same users. History = ratings before the evaluated period (train for
`--split validation`, train + validation for `--split test`); history movies are never recommended; relevant =
ratings ≥ 4.0 in the evaluated period. A fixed-seed sample of `EVAL_SAMPLE_USERS` eligible users is used for
"all" and "regular" users; up to `EVAL_COLD_START_EXTRA` more cold-start users are added for the cold-start
group (1–10 history ratings).

| Metric | Meaning |
|---|---|
| Precision@10, Recall@10, NDCG@10, HitRate@10, MRR@10 | Standard top-10 accuracy |
| Recall@50 | Liked movies in the top 50: for Stage 1, the ceiling for the LLM reranker |
| Coverage@10 | Share of the catalog (movies with earlier ratings) appearing in any top 10 |
| Gini@10 | How unevenly recommendations are spread over the catalog (0 even, 1 concentrated) |
| Serendipity@10 | Liked picks that are not in the 50 most-rated unseen movies, ÷ 10 (Ge et al., 2010) |
| Novelty@10 | Mean −log2(share of users who rated the movie) |

Results are written to `artifacts/evaluation/results.json` (shown on the Compare page, by user group) and
`results_<split>_<group>.csv` (for report tables), with 95% bootstrap intervals for NDCG@10 and Recall@10, paired
NDCG@10 differences against implicit ALS, and NDCG@10 by history length. `evaluate(..., rerank_hook=...)`
scores a reranker over the Stage 1 shortlist with the same protocol; this is how the LLM stage is evaluated.

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

`cd backend && python -m pytest` runs 48 tests covering: the date split boundaries and the per-user split, the
cold-start flag, row cleaning and deduplication, exclusion of already-rated movies in every model, the
proposal's Item-KNN formula on a hand-worked example, explicit ALS lowering its error, the Stage 1 merge rules
(top 30 + top 30, averaging, top-up to 50), NeuMF training and serving (skipped without PyTorch), Gini,
coverage and serendipity on hand-worked values, the full evaluation output, the shortlist export, every LLM
validation rule, the one-retry-then-fallback behaviour, timeouts, that the LLM receives exactly the ALS
top-50, that secrets are not sent or echoed, readiness states, and the streaming endpoints.

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
