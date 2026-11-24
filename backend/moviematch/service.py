"""Readiness checks, loaded artifacts, and recommendation orchestration.

Everything returned from here is computed from the processed MovieLens
artifacts and the saved models. There is no sample or mock data path: if an
artifact is missing, the corresponding feature reports itself unavailable.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from moviematch.config import Settings
from moviematch.data.paths import (
    ARTIFACT_FORMAT_VERSION,
    evaluation_results_path,
    model_paths,
    processed_paths,
)
from moviematch.data.raw import check_raw_dataset
from moviematch.llm.client import OpenAICompatibleClient
from moviematch.llm.reranker import FALLBACK_LABEL, ChatClient, RerankResult, rerank_steps
from moviematch.llm.request_genres import infer_request_genres
from moviematch.models.als import AlsCandidate, AlsModel
from moviematch.models.als_explicit import ExplicitAlsModel
from moviematch.models.common import min_max, top_k, user_row
from moviematch.models.item_knn import Evidence, ItemKnnModel
from moviematch.models.popularity import PopularityModel
from moviematch.models.stage1 import Stage1Candidate, merge_candidates

logger = logging.getLogger("moviematch.service")

MODES = (
    "popularity",
    "popularity_weighted",
    "item_knn",
    "als",
    "als_explicit",
    "stage1",
    "neumf",
    "als_llm",
)
CORE_MODELS = ("popularity", "item_knn", "als")
MODEL_LABELS = {
    "popularity": "Popularity",
    "popularity_weighted": "Popularity (weighted)",
    "item_knn": "Item-KNN",
    "als": "ALS",
    "als_explicit": "ALS (explicit)",
    "stage1": "Stage 1 (merged)",
    "neumf": "NeuMF",
}
TRAIN_HINT = {
    "als_explicit": "Run `python -m moviematch.cli train --only als_explicit`, then restart the backend.",
    "neumf": "Install PyTorch (`pip install torch`), run `python -m moviematch.cli train --only neumf`, then restart the backend.",
}


def _torch_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #

@dataclass
class Readiness:
    state: str
    raw_dataset: dict[str, Any]
    processed: bool
    processed_problem: str | None
    models: dict[str, bool]
    llm: dict[str, Any]
    evaluation_available: bool
    messages: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "rawDataset": self.raw_dataset,
            "processed": self.processed,
            "processedProblem": self.processed_problem,
            "models": self.models,
            "llm": self.llm,
            "evaluationAvailable": self.evaluation_available,
            "messages": self.messages,
        }


def readiness(
    settings: Settings,
    loaded: dict[str, bool] | None = None,
    load_errors: dict[str, str] | None = None,
) -> Readiness:
    """Inspect what exists on disk (and, if given, what is loaded in memory)."""
    raw = check_raw_dataset(settings.movielens_data_path)
    pp, mp = processed_paths(settings), model_paths(settings)

    processed_problem = None
    missing = [p.name for p in pp.all_required() if not p.exists()]
    if missing:
        processed_problem = "Missing processed artifacts: " + ", ".join(missing)
    else:
        try:
            manifest = json.loads(pp.manifest.read_text())
            if manifest.get("artifactFormatVersion") != ARTIFACT_FORMAT_VERSION:
                processed_problem = "Processed artifacts were made by an older version; re-run preprocessing."
        except (OSError, json.JSONDecodeError):
            processed_problem = "manifest.json could not be read; re-run preprocessing."
    processed = processed_problem is None

    on_disk = {
        "popularity": mp.popularity.exists(),
        "popularity_weighted": mp.popularity.exists(),
        "item_knn": mp.knn.exists(),
        "als": mp.als.exists() and mp.als_meta.exists(),
        "als_explicit": mp.als_explicit.exists() and mp.als_explicit_meta.exists(),
        "neumf": mp.neumf.exists() and mp.neumf_meta.exists() and mp.neumf_support.exists(),
    }
    models = {name: bool(on_disk[name] and processed) for name in on_disk}
    if loaded is not None:
        models = {name: bool(models[name] and loaded.get(name, False)) for name in models}
    models["stage1"] = models["als"] and models["item_knn"]
    models["als_llm"] = models["stage1"] and settings.llm.configured

    llm = {
        "enabled": settings.llm.enabled,
        "configured": settings.llm.configured,
        "model": settings.llm.model if settings.llm.configured else None,
        "baseUrlSet": bool(settings.llm.base_url),
        "apiKeySet": bool(settings.llm.api_key),
        "timeoutSeconds": settings.llm.timeout_seconds,
        "jsonMode": settings.llm.json_mode,
        "missing": settings.llm.missing,
    }

    messages: list[dict[str, str]] = []
    if not processed:
        if raw.ok:
            state = "not_processed"
            messages.append({
                "level": "warning",
                "code": "dataset_not_processed",
                "message": "MovieLens 32M was found but has not been preprocessed.",
                "action": "Run the preprocessing command, then train the models and restart the backend.",
            })
        else:
            state = "dataset_unavailable"
            messages.append({
                "level": "error",
                "code": "dataset_unavailable",
                "message": "MovieLens 32M was not found. " + (raw.problem() or ""),
                "action": "Set MOVIELENS_DATA_PATH and run the preprocessing command.",
            })
    elif not any(on_disk[n] for n in CORE_MODELS):
        state = "models_not_trained"
        messages.append({
            "level": "warning",
            "code": "models_not_trained",
            "message": "The dataset is ready, but no models have been trained.",
            "action": "Run the model-training command, then restart the backend.",
        })
    else:
        state = "ready" if all(models[n] for n in CORE_MODELS) else "partial"
        labels = {n: MODEL_LABELS[n] for n in CORE_MODELS}
        for name, label in labels.items():
            if not on_disk[name]:
                messages.append({
                    "level": "warning",
                    "code": f"{name}_unavailable",
                    "message": f"The dataset is ready, but the {label} model has not been trained.",
                    "action": f"Run the model-training command before requesting {label} recommendations.",
                })
            elif not models[name] and load_errors and name in load_errors:
                messages.append({
                    "level": "warning",
                    "code": f"{name}_load_failed",
                    "message": f"The saved {label} model could not be loaded ({load_errors[name]}).",
                    "action": "Re-run the model-training command, then restart the backend.",
                })
            elif not models[name]:
                messages.append({
                    "level": "warning",
                    "code": f"{name}_not_loaded",
                    "message": f"The {label} model exists on disk but is not loaded.",
                    "action": "Restart the backend to load newly trained models.",
                })

        for name in ("als_explicit", "neumf"):
            if models[name]:
                continue
            if load_errors and name in load_errors:
                text = f"The saved {MODEL_LABELS[name]} model could not be loaded ({load_errors[name]})."
            elif on_disk[name] and loaded is not None:
                text = f"The {MODEL_LABELS[name]} model exists on disk but is not loaded."
            else:
                text = f"The optional {MODEL_LABELS[name]} model has not been trained."
            messages.append({
                "level": "info",
                "code": f"{name}_unavailable",
                "message": text,
                "action": TRAIN_HINT[name],
            })

    if not settings.llm.configured:
        messages.append({
            "level": "info",
            "code": "llm_not_configured",
            "message": "LLM reranking is not configured.",
            "action": "Set LLM_ENABLED=true, LLM_BASE_URL, and LLM_MODEL (and LLM_API_KEY if your provider needs one), then restart the backend.",
        })

    return Readiness(
        state=state,
        raw_dataset={
            "configured": raw.configured,
            "found": raw.ok,
            "problem": raw.problem(),
        },
        processed=processed,
        processed_problem=processed_problem,
        models=models,
        llm=llm,
        evaluation_available=evaluation_results_path(settings).exists(),
        messages=messages,
    )


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #

def _iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()


class MovieMatchService:
    def __init__(self, settings: Settings, llm_client: ChatClient | None = None):
        self.settings = settings
        self.loaded = {name: False for name in ("processed", *MODEL_LABELS)}
        self.load_errors: dict[str, str] = {}
        self._llm_client = llm_client
        self.popularity: PopularityModel | None = None
        self.popularity_weighted: PopularityModel | None = None
        self.knn: ItemKnnModel | None = None
        self.als: AlsModel | None = None
        self.als_explicit: ExplicitAlsModel | None = None
        self.neumf = None

    # ---------------------------- loading ---------------------------------- #

    def load(self) -> None:
        settings = self.settings
        state = readiness(settings)
        if state.processed:
            try:
                self._load_processed()
                self.loaded["processed"] = True
            except Exception as exc:  # noqa: BLE001 - reported via status
                logger.exception("Failed to load processed artifacts")
                self.load_errors["processed"] = type(exc).__name__
        if self.loaded["processed"]:
            mp = model_paths(settings)
            def load_neumf():
                if not _torch_installed():
                    raise ModuleNotFoundError("PyTorch not installed")
                from moviematch.models.neumf import NeuMFModel

                return NeuMFModel.load(mp.neumf, mp.neumf_meta)

            loaders = {
                "popularity": lambda: PopularityModel.load(mp.popularity, "count"),
                "popularity_weighted": lambda: PopularityModel.load(mp.popularity, "weighted"),
                "item_knn": lambda: ItemKnnModel.load(
                    mp.knn, settings.high_rating_threshold, settings.knn_scoring, settings.knn_min_support
                ),
                "als": lambda: AlsModel.load(mp.als, mp.als_meta),
                "als_explicit": lambda: ExplicitAlsModel.load(mp.als_explicit, mp.als_explicit_meta),
                "neumf": load_neumf,
            }
            for name, loader in loaders.items():
                if not state.models.get(name):
                    continue
                try:
                    setattr(self, "knn" if name == "item_knn" else name, loader())
                    self.loaded[name] = True
                except ModuleNotFoundError:
                    self.load_errors[name] = "PyTorch not installed"
                except FileNotFoundError as exc:
                    logger.warning("%s: %s", name, exc)
                    self.load_errors[name] = "trained by an older version; retrain"
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to load %s model", name)
                    self.load_errors[name] = type(exc).__name__
            if self.als is not None and self.als.n_items != self.n_movies:
                logger.error("ALS model does not match the processed dataset; retrain it.")
                self.als, self.loaded["als"] = None, False
                self.load_errors["als"] = "dataset mismatch"
            if self.als_explicit is not None and self.als_explicit.n_items != self.n_movies:
                self.als_explicit, self.loaded["als_explicit"] = None, False
                self.load_errors["als_explicit"] = "dataset mismatch"
            if self.knn is not None and self.knn.similarity.shape[0] != self.n_movies:
                logger.error("Item-KNN model does not match the processed dataset; retrain it.")
                self.knn, self.loaded["item_knn"] = None, False
                self.load_errors["item_knn"] = "dataset mismatch"
        if settings.llm.configured and self._llm_client is None:
            self._llm_client = OpenAICompatibleClient(settings.llm)

    def _load_processed(self) -> None:
        pp = processed_paths(self.settings)
        self.manifest = json.loads(pp.manifest.read_text())
        self.statistics = json.loads(pp.statistics.read_text())
        self.genre_vocabulary: list[str] = self.manifest["config"]["genreVocabulary"]

        movies = pd.read_parquet(pp.movies)
        self.n_movies = len(movies)
        self.movie_ids = movies["movieId"].to_numpy(dtype=np.int64)
        self.movie_titles = movies["title"].astype(str).tolist()
        self.movie_years = [None if pd.isna(y) else int(y) for y in movies["year"]]
        self.movie_genres = [g.split("|") if g else [] for g in movies["genres"].astype(str)]
        self.genre_masks = {
            genre: np.fromiter((genre in genres for genres in self.movie_genres), dtype=bool, count=len(self.movie_genres))
            for genre in self.genre_vocabulary
        }
        self.movie_imdb = [None if pd.isna(v) else str(v) for v in movies["imdbId"]]
        self.movie_tmdb = [None if pd.isna(v) else int(v) for v in movies["tmdbId"]]
        self.movie_rating_count = movies["rating_count"].to_numpy()
        self.movie_mean_rating = movies["mean_rating"].to_numpy()

        tags = pd.read_parquet(pp.movie_tags)
        self.movie_tags: dict[int, list[str]] = {}
        for idx, tag in zip(tags["movie_idx"].to_numpy(), tags["tag"].astype(str)):
            self.movie_tags.setdefault(int(idx), []).append(tag)

        users = pd.read_parquet(pp.users)
        self.user_ids = users["userId"].to_numpy(dtype=np.int64)
        self.user_id_str = users["userId"].astype(str).to_numpy(dtype=str)
        self.user_rating_count = users["rating_count"].to_numpy()
        self.user_train_count = users["train_count"].to_numpy()
        self.user_validation_count = (
            users["validation_count"].to_numpy() if "validation_count" in users else np.zeros(len(users), int)
        )
        self.user_is_cold = (
            users["is_cold_start"].to_numpy() if "is_cold_start" in users else np.zeros(len(users), bool)
        )
        self.user_mean_rating = users["mean_rating"].to_numpy()
        self.user_first_ts = users["first_timestamp"].to_numpy()
        self.user_last_ts = users["last_timestamp"].to_numpy()
        self.user_favorite_genres = [g.split("|") if g else [] for g in users["favorite_genres"].astype(str)]

        top = pd.read_parquet(pp.user_top_movies).sort_values(["user_idx", "rank"])
        self.top_user = top["user_idx"].to_numpy()
        self.top_movie = top["movie_idx"].to_numpy()
        self.top_rating = top["rating"].to_numpy()

        self.full = sp.load_npz(pp.full_matrix).tocsr()

    # ------------------------------ status --------------------------------- #

    def readiness(self) -> Readiness:
        return readiness(self.settings, loaded=self.loaded, load_errors=self.load_errors)

    def require_processed(self) -> None:
        if not self.loaded["processed"]:
            state = self.readiness()
            first = state.messages[0] if state.messages else None
            raise ApiError(
                503,
                state.state if state.state != "ready" else "dataset_not_loaded",
                (first["message"] + " " + first["action"]) if first else "Dataset is not loaded.",
            )

    # ------------------------------ lookups -------------------------------- #

    def movie_index(self, movie_id: int) -> int:
        pos = int(np.searchsorted(self.movie_ids, movie_id))
        if pos >= len(self.movie_ids) or self.movie_ids[pos] != movie_id:
            raise ApiError(404, "movie_not_found", f"MovieLens movie {movie_id} does not exist.")
        return pos

    def user_index(self, user_id: int) -> int:
        pos = int(np.searchsorted(self.user_ids, user_id))
        if pos >= len(self.user_ids) or self.user_ids[pos] != user_id:
            raise ApiError(404, "user_not_found", f"MovieLens user {user_id} was not found.")
        return pos

    def poster_url(self, idx: int) -> str | None:
        template = self.settings.poster_url_template
        if not template:
            return None
        values = {
            "movieId": int(self.movie_ids[idx]),
            "imdbId": self.movie_imdb[idx],
            "tmdbId": self.movie_tmdb[idx],
        }
        try:
            needed = [k for k in values if "{" + k + "}" in template]
            if any(values[k] is None for k in needed):
                return None
            return template.format(**values)
        except (KeyError, IndexError, ValueError):
            return None

    def movie_out(self, idx: int, tag_limit: int = 10) -> dict[str, Any]:
        return {
            "movieId": int(self.movie_ids[idx]),
            "title": self.movie_titles[idx],
            "year": self.movie_years[idx],
            "genres": self.movie_genres[idx],
            "tags": self.movie_tags.get(idx, [])[:tag_limit],
            "posterUrl": self.poster_url(idx),
        }

    def movie_details(self, movie_id: int) -> dict[str, Any]:
        self.require_processed()
        idx = self.movie_index(movie_id)
        out = self.movie_out(idx, tag_limit=self.settings.tags_per_movie)
        out.update({
            "ratingCount": int(self.movie_rating_count[idx]),
            "meanRating": round(float(self.movie_mean_rating[idx]), 3) if self.movie_rating_count[idx] else None,
            "imdbId": self.movie_imdb[idx],
            "tmdbId": self.movie_tmdb[idx],
        })
        return out

    def statistics_out(self) -> dict[str, Any]:
        self.require_processed()
        stats = dict(self.statistics)
        featured_idx = stats.pop("mostRatedMovieIdx", [])[:12]
        stats["mostRatedMovies"] = [
            {**self.movie_out(i, tag_limit=3), "ratingCount": int(self.movie_rating_count[i])}
            for i in featured_idx
        ]
        stats["processedAt"] = self.manifest.get("createdAt")
        stats["matchesPublishedMl32mCounts"] = self.manifest["dataset"].get("matchesPublishedMl32mCounts")
        return stats

    def search_users(self, query: str, page: int, page_size: int) -> dict[str, Any]:
        self.require_processed()
        q = query.strip()
        if q and not q.isdigit():
            raise ApiError(400, "invalid_query", "Search by numeric MovieLens user ID.")
        if q:
            matches = np.flatnonzero(np.char.startswith(self.user_id_str, q))
            exact = matches[self.user_ids[matches] == int(q)]
            rest = matches[self.user_ids[matches] != int(q)]
            matches = np.concatenate((exact, rest))
        else:
            matches = np.arange(len(self.user_ids))
        total = int(len(matches))
        start = (page - 1) * page_size
        chunk = matches[start : start + page_size]
        return {
            "query": q,
            "page": page,
            "pageSize": page_size,
            "total": total,
            "totalPages": max(1, -(-total // page_size)),
            "users": [
                {
                    "userId": int(self.user_ids[u]),
                    "ratingCount": int(self.user_rating_count[u]),
                    "favoriteGenres": self.user_favorite_genres[u],
                }
                for u in chunk
            ],
        }

    def _highly_rated(self, user_idx: int) -> list[dict[str, Any]]:
        lo = int(np.searchsorted(self.top_user, user_idx, side="left"))
        hi = int(np.searchsorted(self.top_user, user_idx, side="right"))
        return [
            {
                "movieId": int(self.movie_ids[m]),
                "title": self.movie_titles[m],
                "year": self.movie_years[m],
                "rating": float(r),
            }
            for m, r in zip(self.top_movie[lo:hi], self.top_rating[lo:hi])
        ]

    def user_summary(self, user_id: int) -> dict[str, Any]:
        self.require_processed()
        u = self.user_index(user_id)
        return {
            "userId": int(self.user_ids[u]),
            "ratingCount": int(self.user_rating_count[u]),
            "trainingRatingCount": int(self.user_train_count[u]),
            "validationRatingCount": int(self.user_validation_count[u]),
            "isColdStart": bool(self.user_is_cold[u]),
            "coldStartMaxRatings": self.settings.cold_start_max_ratings,
            "splitMode": self.manifest["config"].get("splitMode", "per_user"),
            "meanRating": round(float(self.user_mean_rating[u]), 3),
            "firstRatingAt": _iso(self.user_first_ts[u]),
            "lastRatingAt": _iso(self.user_last_ts[u]),
            "favoriteGenres": self.user_favorite_genres[u],
            "highlyRatedMovies": self._highly_rated(u),
            "highRatingThreshold": self.settings.high_rating_threshold,
        }

    # -------------------------- recommendations ---------------------------- #

    def _evidence_out(self, items: list[Evidence]) -> list[dict[str, Any]]:
        return [
            {
                "historyMovieId": int(self.movie_ids[e.history_movie_idx]),
                "historyMovieTitle": self.movie_titles[e.history_movie_idx],
                "similarity": round(e.similarity, 4),
                "historyRating": e.history_rating,
            }
            for e in items
        ]

    def _evidence_sentence(self, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return ""
        picks = evidence[:2]
        titles = " and ".join(e["historyMovieTitle"] for e in picks)
        ratings = " and ".join(f"{e['historyRating']:.1f}" for e in picks)
        return f"Similar to {titles}, which you rated {ratings}."

    def _require_model(self, name: str) -> None:
        if not self.loaded.get(name):
            hint = TRAIN_HINT.get(name, "Run the model-training command, then restart the backend.")
            reason = self.load_errors.get(name)
            raise ApiError(
                503,
                "model_unavailable",
                f"The {MODEL_LABELS.get(name, name)} model is not available"
                + (f" ({reason})" if reason else "")
                + f". {hint}",
            )

    def run(
        self,
        mode: str,
        user_id: int,
        top_n: int,
        candidate_count: int = 50,
        user_request: str = "",
    ) -> Iterator[dict[str, Any]]:
        """Yield progress events and finally a ``result`` event."""
        if mode not in MODES:
            raise ApiError(400, "invalid_mode", f"Unknown mode {mode}.")
        self.require_processed()
        required = {
            "popularity": ["popularity"],
            "popularity_weighted": ["popularity_weighted"],
            "item_knn": ["item_knn"],
            "als": ["als"],
            "als_explicit": ["als_explicit"],
            "stage1": ["als", "item_knn"],
            "neumf": ["neumf"],
            "als_llm": ["als", "item_knn"],
        }[mode]
        for name in required:
            self._require_model(name)
        if mode == "als_llm" and (not self.settings.llm.configured or self._llm_client is None):
            raise ApiError(409, "llm_not_configured", "LLM reranking is not configured.")
        top_n = min(top_n, candidate_count) if mode in ("als", "als_explicit", "als_llm") else top_n
        u = self.user_index(user_id)  # validate before any progress events
        return self._run(mode, u, user_id, top_n, candidate_count, user_request)

    def _run(
        self,
        mode: str,
        u: int,
        user_id: int,
        top_n: int,
        candidate_count: int,
        user_request: str,
    ) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        timings: dict[str, float] = {}

        def stage(name: str, status: str, **details: Any) -> dict[str, Any]:
            return {"type": "stage", "stage": name, "status": status, **details}

        # 1. Validate user and load history.
        yield stage("loading_history", "started")
        t0 = time.perf_counter()
        history, history_ratings = user_row(self.full, u)
        timings["loadingHistory"] = (time.perf_counter() - t0) * 1000
        yield stage("loading_history", "done", ratingCount=int(len(history)))

        recommendations: list[dict[str, Any]] = []
        response: dict[str, Any] = {
            "mode": mode,
            "userId": int(user_id),
            "topN": top_n,
            "usedFallback": False,
            "fallbackLabel": None,
            "fallbackReason": None,
            "interpretedRequest": None,
            "llm": None,
        }

        if mode in ("popularity", "popularity_weighted"):
            yield stage("ranking_popularity", "started")
            t0 = time.perf_counter()
            model = self.popularity if mode == "popularity" else self.popularity_weighted
            assert model is not None
            for rank, item in enumerate(model.recommend(history, top_n), start=1):
                if mode == "popularity":
                    text = (
                        f"One of the most-rated movies: {item.rating_count:,} training ratings, "
                        f"averaging {item.mean_rating:.2f}/5."
                    )
                else:
                    text = (
                        ("Widely and well rated: " if item.rating_count >= model.min_votes else "")
                        + f"{item.rating_count:,} training ratings averaging "
                        f"{item.mean_rating:.2f}/5 (weighted rating {item.weighted_rating:.2f})."
                    )
                recommendations.append({
                    "movie": self.movie_out(item.movie_idx),
                    "mode": mode,
                    "finalRank": rank,
                    "popularityScore": round(item.weighted_rating, 4),
                    "ratingCount": item.rating_count,
                    "meanRating": round(item.mean_rating, 3),
                    "explanation": text,
                    "explanationSource": "template",
                    "usedFallback": False,
                })
            timings["popularity"] = (time.perf_counter() - t0) * 1000
            yield stage("ranking_popularity", "done")

        elif mode == "item_knn":
            yield stage("finding_similar", "started")
            t0 = time.perf_counter()
            assert self.knn is not None
            picks = self.knn.recommend(history, history_ratings, top_n)
            evidence = self.knn.evidence(history, history_ratings, [i for i, _ in picks])
            average = self.knn.scoring == "average"
            response["knnScoring"] = self.knn.scoring
            for rank, (idx, score) in enumerate(picks, start=1):
                ev = self._evidence_out(evidence.get(idx, []))
                lead = f"Predicted rating {min(score, 5.0):.1f}/5 from similar movies you rated. " if average else ""
                recommendations.append({
                    "movie": self.movie_out(idx),
                    "mode": mode,
                    "finalRank": rank,
                    "knnScore": round(score, 4),
                    "knnEvidence": ev,
                    "explanation": (lead + (self._evidence_sentence(ev) or "Similar to movies in your rating history.")).strip(),
                    "explanationSource": "template",
                    "usedFallback": False,
                })
            timings["itemKnn"] = (time.perf_counter() - t0) * 1000
            yield stage("finding_similar", "done")

        elif mode == "als_explicit":
            yield stage("generating_als", "started")
            t0 = time.perf_counter()
            assert self.als_explicit is not None
            pool = self.als_explicit.candidates(history, history_ratings, max(top_n, 1))
            timings["alsExplicit"] = (time.perf_counter() - t0) * 1000
            yield stage("generating_als", "done", candidates=len(pool))
            yield stage("finding_similar", "started")
            evidence = self.knn.evidence(history, history_ratings, [c.movie_idx for c in pool]) if self.knn else {}
            yield stage("finding_similar", "done", available=self.knn is not None)
            for c in pool:
                ev = self._evidence_out(evidence.get(c.movie_idx, []))
                recommendations.append({
                    "movie": self.movie_out(c.movie_idx),
                    "mode": mode,
                    "finalRank": c.rank,
                    "originalAlsRank": c.rank,
                    "alsScore": round(c.normalized_score, 4),
                    "alsRawScore": round(c.raw_score, 5),
                    "predictedRating": round(c.raw_score, 3),
                    "knnEvidence": ev,
                    "strongestKnnSimilarity": ev[0]["similarity"] if ev else None,
                    "explanation": (
                        f"Explicit ALS predicts you would rate this {min(c.raw_score, 5.0):.1f}/5. "
                        + self._evidence_sentence(ev)
                    ).strip(),
                    "explanationSource": "template",
                    "usedFallback": False,
                })

        elif mode in ("stage1", "als_llm"):
            yield stage("generating_als", "started")
            t0 = time.perf_counter()
            assert self.als is not None and self.knn is not None
            als_scores = self.als.scores_for_vector(self.als.user_vector(history, history_ratings))
            timings["als"] = (time.perf_counter() - t0) * 1000
            yield stage("generating_als", "done")
            yield stage("finding_similar", "started")
            t0 = time.perf_counter()
            knn_scores = self.knn.scores(history, history_ratings)
            timings["itemKnn"] = (time.perf_counter() - t0) * 1000
            yield stage("finding_similar", "done")
            if mode == "als_llm":
                als_scores, knn_scores, request_filter = self._apply_request_filter(
                    als_scores, knn_scores, history, user_request,
                    min(candidate_count, self.settings.stage1_pool_size),
                )
                if request_filter:
                    response["requestFilter"] = request_filter
            yield stage("merging_candidates", "started")
            t0 = time.perf_counter()
            pool1: list[Stage1Candidate] = merge_candidates(
                als_scores, knn_scores, history,
                self.settings.stage1_per_model,
                min(candidate_count, self.settings.stage1_pool_size) if mode == "als_llm" else self.settings.stage1_pool_size,
                self.settings.stage1_top_up,
            )
            evidence = self.knn.evidence(history, history_ratings, [c.movie_idx for c in pool1])
            timings["merge"] = (time.perf_counter() - t0) * 1000
            response["candidateCount"] = len(pool1)
            response["candidateSources"] = {
                src: sum(1 for c in pool1 if c.source == src) for src in ("both", "als", "knn", "als_top_up")
            }
            yield stage("merging_candidates", "done", candidates=len(pool1))
            per = self.settings.stage1_per_model
            def stage1_item(c: Stage1Candidate, final_rank: int) -> dict[str, Any]:
                ev = self._evidence_out(evidence.get(c.movie_idx, []))
                where = {
                    "both": f"In both ALS's and Item-KNN's top {per}",
                    "als": f"In ALS's top {per}",
                    "knn": f"In Item-KNN's top {per}",
                    "als_top_up": "Added from ALS's next ranks to fill the shortlist",
                }[c.source]
                return {
                    "movie": self.movie_out(c.movie_idx),
                    "mode": mode,
                    "finalRank": final_rank,
                    "stage1Rank": c.rank,
                    "mergedScore": round(c.merged_score, 4),
                    "alsScore": None if c.als_score is None else round(c.als_score, 4),
                    "knnScore": None if c.knn_score is None else round(c.knn_score, 4),
                    "originalAlsRank": c.als_rank,
                    "knnRank": c.knn_rank,
                    "candidateSource": c.source,
                    "knnEvidence": ev,
                    "strongestKnnSimilarity": ev[0]["similarity"] if ev else None,
                    "explanation": (f"{where}; merged score {c.merged_score:.2f}. " + self._evidence_sentence(ev)).strip(),
                    "explanationSource": "template",
                    "usedFallback": False,
                }

            if mode == "stage1":
                recommendations = [stage1_item(c, c.rank) for c in pool1[:top_n]]
            else:
                payload = self._llm_payload(u, user_request, pool1, evidence)
                assert self._llm_client is not None
                t0 = time.perf_counter()
                steps = rerank_steps(self._llm_client, payload, top_n, self.genre_vocabulary)
                while True:
                    try:
                        name, status, details = next(steps)
                    except StopIteration as stop:
                        result: RerankResult = stop.value
                        break
                    yield stage(name, status, **details)
                timings["llm"] = (time.perf_counter() - t0) * 1000
                response["llm"] = {"attempts": result.attempts, "model": self.settings.llm.model}

                if result.succeeded:
                    assert result.output is not None
                    by_id = {int(self.movie_ids[c.movie_idx]): c for c in pool1}
                    for ranked in result.output.rankedMovies:
                        item = stage1_item(by_id[ranked.movieId], ranked.rank)
                        item["llmRelevanceScore"] = round(ranked.llmRelevanceScore, 4)
                        item["explanation"] = ranked.explanation
                        item["explanationSource"] = "template"
                        recommendations.append(item)
                    response["interpretedRequest"] = result.output.interpretedRequest.model_dump()
                else:
                    response.update({
                        "usedFallback": True,
                        "fallbackLabel": FALLBACK_LABEL,
                        "fallbackReason": result.fallback_reason,
                    })
                    for c in pool1[:top_n]:
                        item = stage1_item(c, c.rank)
                        item["usedFallback"] = True
                        recommendations.append(item)

        elif mode == "neumf":
            yield stage("scoring_neumf", "started")
            t0 = time.perf_counter()
            assert self.neumf is not None
            if self.neumf.knows(u):
                scores = self.neumf.scores(u)
                picks = top_k(scores, max(top_n, 1), exclude=history)
                norm = min_max(scores[picks]) if len(picks) else np.zeros(0)
                for rank, idx in enumerate(picks, start=1):
                    recommendations.append({
                        "movie": self.movie_out(int(idx)),
                        "mode": mode,
                        "finalRank": rank,
                        "neumfScore": round(float(norm[rank - 1]), 4),
                        "explanation": (
                            f"NeuMF ranked this #{rank} for you from patterns it learned across all users' ratings."
                        ),
                        "explanationSource": "template",
                        "usedFallback": False,
                    })
            else:
                response.update({
                    "usedFallback": True,
                    "fallbackLabel": "Most-rated movies, because NeuMF has no training data for this user.",
                    "fallbackReason": "This user has no ratings in the data NeuMF was trained on.",
                })
                if self.popularity is not None:
                    for rank, item in enumerate(self.popularity.recommend(history, top_n), start=1):
                        recommendations.append({
                            "movie": self.movie_out(item.movie_idx),
                            "mode": mode,
                            "finalRank": rank,
                            "ratingCount": item.rating_count,
                            "explanation": f"One of the most-rated movies: {item.rating_count:,} training ratings.",
                            "explanationSource": "template",
                            "usedFallback": True,
                        })
            timings["neumf"] = (time.perf_counter() - t0) * 1000
            yield stage("scoring_neumf", "done")

        else:
            # 2. ALS candidates.
            yield stage("generating_als", "started")
            t0 = time.perf_counter()
            assert self.als is not None
            pool: list[AlsCandidate] = self.als.candidates(history, history_ratings, candidate_count)
            timings["als"] = (time.perf_counter() - t0) * 1000
            yield stage("generating_als", "done", candidates=len(pool))
            response["candidateCount"] = len(pool)

            # 3. KNN similarity evidence (when the KNN model is available).
            yield stage("finding_similar", "started")
            t0 = time.perf_counter()
            evidence: dict[int, list[Evidence]] = {}
            if self.knn is not None:
                evidence = self.knn.evidence(history, history_ratings, [c.movie_idx for c in pool])
            timings["knnEvidence"] = (time.perf_counter() - t0) * 1000
            yield stage("finding_similar", "done", available=self.knn is not None)

            def als_item(c: AlsCandidate, final_rank: int) -> dict[str, Any]:
                ev = self._evidence_out(evidence.get(c.movie_idx, []))
                return {
                    "movie": self.movie_out(c.movie_idx),
                    "mode": mode,
                    "finalRank": final_rank,
                    "originalAlsRank": c.rank,
                    "alsScore": round(c.normalized_score, 4),
                    "alsRawScore": round(c.raw_score, 5),
                    "knnEvidence": ev,
                    "strongestKnnSimilarity": ev[0]["similarity"] if ev else None,
                    "explanation": (
                        f"ALS ranked this #{c.rank} of the {len(pool)} unseen candidates for your rating history. "
                        + self._evidence_sentence(ev)
                    ).strip(),
                    "explanationSource": "template",
                    "usedFallback": False,
                }

            recommendations = [als_item(c, c.rank) for c in pool[:top_n]]

        response["recommendations"] = recommendations
        response["stageTimingsMs"] = {k: round(v, 1) for k, v in timings.items()}
        response["elapsedMs"] = round((time.perf_counter() - started) * 1000, 1)
        if not recommendations:
            response["message"] = "No recommendations could be produced for this user."
        yield {"type": "result", "data": response}

    def run_to_completion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = None
        for event in self.run(*args, **kwargs):
            if event["type"] == "result":
                result = event["data"]
        assert result is not None
        return result

    def _apply_request_filter(
        self,
        als_scores: np.ndarray,
        knn_scores: np.ndarray,
        history: np.ndarray,
        user_request: str,
        pool_size: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
        """Draw the shortlist from movies that fit the request's genres.

        Stage 1 scores come from the rating history alone, so without this a
        request like "romantic" can only reorder movies ALS already picked.
        Movies outside the requested genres get -inf, so they cannot enter the
        pool; everything else (ALS/KNN scores, exclusion of rated movies) is
        unchanged. If the request names several genres, movies matching all of
        them are used when there are enough, otherwise any of them.
        """
        genres = infer_request_genres(user_request, self.genre_vocabulary)
        if not genres:
            return als_scores, knn_scores, None
        unseen = np.ones(self.n_movies, dtype=bool)
        unseen[history] = False
        has_all = np.logical_and.reduce([self.genre_masks[g] for g in genres])
        has_any = np.logical_or.reduce([self.genre_masks[g] for g in genres])
        match_all = len(genres) == 1 or int((has_all & unseen & np.isfinite(als_scores)).sum()) >= pool_size
        keep = has_all if match_all else has_any
        filtered_als = np.where(keep, als_scores, -np.inf)
        filtered_knn = np.where(keep, knn_scores, -np.inf)
        info = {
            "genres": genres,
            "match": "all" if match_all else "any",
            "eligibleMovies": int((keep & unseen).sum()),
        }
        return filtered_als, filtered_knn, info

    def _llm_payload(
        self,
        user_idx: int,
        user_request: str,
        pool: list[Stage1Candidate],
        evidence: dict[int, list[Evidence]],
    ) -> dict[str, Any]:
        """Compact, grounded input: only data the backend can vouch for."""
        profile_movies = [
            {"movieId": m["movieId"], "title": m["title"], "rating": m["rating"]}
            for m in self._highly_rated(user_idx)
        ]
        candidates = []
        for c in pool:
            idx = c.movie_idx
            entry: dict[str, Any] = {
                "movieId": int(self.movie_ids[idx]),
                "title": self.movie_titles[idx],
                "genres": self.movie_genres[idx],
                "tags": self.movie_tags.get(idx, [])[:6],
                "mergedScore": round(c.merged_score, 3),
                "stage1Rank": c.rank,
                "alsScore": None if c.als_score is None else round(c.als_score, 3),
                "knnScore": None if c.knn_score is None else round(c.knn_score, 3),
                "originalAlsRank": c.als_rank,
                "candidateSource": c.source,
                "knnEvidence": [
                    {"historyMovie": self.movie_titles[e.history_movie_idx], "similarity": round(e.similarity, 3)}
                    for e in evidence.get(idx, [])[:2]
                ],
            }
            if self.movie_years[idx] is not None:
                entry["year"] = self.movie_years[idx]
            candidates.append(entry)
        return {
            "userRequest": user_request.strip(),
            "userProfile": {
                "userId": int(self.user_ids[user_idx]),
                "highlyRatedMovies": profile_movies,
                "preferredGenres": self.user_favorite_genres[user_idx],
            },
            "candidates": candidates,
        }

    # ---------------------------- evaluation ------------------------------- #

    def evaluation_results(self) -> dict[str, Any]:
        path = evaluation_results_path(self.settings)
        if not path.exists():
            return {"available": False, "message": "Evaluation not run yet."}
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {"available": False, "message": "Evaluation results could not be read."}
