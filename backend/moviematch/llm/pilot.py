"""Small, reproducible LLM comparison on the validation period."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from moviematch.config import Settings
from moviematch.data.paths import processed_paths
from moviematch.llm.client import OpenAICompatibleClient
from moviematch.llm.reranker import rerank
from moviematch.models.candidates import export_candidates
from moviematch.models.evaluation import ranking_metrics


def _ranked_ids(frame: pd.DataFrame, column: str) -> list[int]:
    ordered = frame[frame[column].notna()].sort_values(column)
    return ordered["movie_id"].astype(int).tolist()[:10]


def run_validation_pilot(settings: Settings, users: int = 5, log=print) -> Path:
    if not settings.llm.configured:
        raise RuntimeError("Configure an LLM before running the pilot.")
    if settings.split_mode != "date":
        raise RuntimeError("The validation pilot requires SPLIT_MODE=date.")
    if users < 1:
        raise ValueError("users must be at least 1")

    export = export_candidates(settings, split_name="validation", n_users=users, log=log)
    frame = pd.read_parquet(export["candidates"])
    profiles = [json.loads(line) for line in Path(export["userFile"]).read_text().splitlines()]
    manifest = json.loads(processed_paths(settings).manifest.read_text())
    genres = manifest["config"]["genreVocabulary"]
    grouped = {int(user_id): rows.sort_values("rank") for user_id, rows in frame.groupby("user_id")}

    records = []
    metrics = defaultdict(lambda: defaultdict(list))
    client = OpenAICompatibleClient(settings.llm)
    try:
        for index, profile in enumerate(profiles, start=1):
            user_id = profile["user_id"]
            candidates = grouped[user_id]
            stage1 = candidates["movie_id"].astype(int).tolist()
            als = _ranked_ids(candidates, "als_rank")
            knn = _ranked_ids(candidates, "knn_rank")
            payload = {
                "userRequest": "",
                "userProfile": {
                    "userId": user_id,
                    "highlyRatedMovies": [
                        {"movieId": movie["movie_id"], "title": movie["title"], "rating": movie["rating"]}
                        for movie in profile["highly_rated"]
                    ],
                    "preferredGenres": profile["favorite_genres"],
                    "recentlyRatedMovies": profile["recent_5"],
                },
                "candidates": [
                    {
                        "movieId": int(row.movie_id),
                        "title": row.title,
                        "year": None if pd.isna(row.year) else int(row.year),
                        "genres": list(row.genres),
                        "tags": list(row.tags),
                        "mergedScore": float(row.merged_score),
                        "stage1Rank": int(row.rank),
                        "alsScore": None if pd.isna(row.als_score) else float(row.als_score),
                        "knnScore": None if pd.isna(row.knn_score) else float(row.knn_score),
                        "knnEvidence": json.loads(row.knn_evidence),
                    }
                    for row in candidates.itertuples()
                ],
            }
            result = rerank(client, payload, 10, genres)
            llm = [movie.movieId for movie in result.output.rankedMovies] if result.output else stage1[:10]
            relevant = set(profile["relevant_movie_ids"])
            user_metrics = {}
            for name, ids in {
                "stage1": stage1[:10], "als": als,
                f"item_knn_{settings.knn_scoring}": knn, "llm_or_fallback": llm,
            }.items():
                score = ranking_metrics(np.asarray(ids, dtype=np.int64), relevant, 10)
                user_metrics[name] = {metric + "@10": round(float(score[metric]), 5) for metric in ("ndcg", "recall", "precision", "hitRate")}
                for metric in ("ndcg", "recall", "precision", "hitRate"):
                    metrics[name][metric].append(float(score[metric]))

            explanations = []
            if result.output:
                by_id = {movie["movieId"]: movie for movie in payload["candidates"]}
                explanations = [
                    {
                        "movieId": movie.movieId,
                        "title": by_id[movie.movieId]["title"],
                        "genres": by_id[movie.movieId]["genres"],
                        "tags": by_id[movie.movieId]["tags"],
                        "explanation": movie.explanation,
                    }
                    for movie in result.output.rankedMovies
                ]
            records.append({
                "userId": user_id,
                "coldStart": profile["is_cold_start"],
                "historySize": profile["history_size"],
                "relevantCount": len(relevant),
                "usedFallback": not result.succeeded,
                "attempts": result.attempts,
                "fallbackReason": result.fallback_reason,
                "validationErrors": result.validation_errors,
                "metrics": user_metrics,
                "top10": {"stage1": stage1[:10], "als": als, "itemKnn": knn, "llmOrFallback": llm},
                "explanations": explanations,
            })
            log(f"  {index}/{len(profiles)} user {user_id}: {'LLM' if result.succeeded else 'fallback'}")
    finally:
        client.close()

    report = {
        "kind": "exploratory_validation_pilot",
        "status": "small sample; not a final test-set evaluation",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "model": settings.llm.model,
        "users": len(records),
        "coldStartUsers": sum(record["coldStart"] for record in records),
        "llmSuccesses": sum(not record["usedFallback"] for record in records),
        "llmFallbacks": sum(record["usedFallback"] for record in records),
        "metricMeans": {
            name: {metric + "@10": round(float(np.mean(values)), 5) for metric, values in scores.items()}
            for name, scores in metrics.items()
        },
        "notes": [
            "All methods use the same users and held-out validation ratings.",
            "The LLM receives only pre-validation user ratings and tags available before validation.",
            "The LLM-or-fallback metrics include Stage 1 fallback when an LLM response is invalid.",
            "This small pilot does not establish statistical superiority or explanation faithfulness.",
        ],
        "records": records,
    }
    output = settings.artifacts_dir / "llm_pilot" / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    log(f"Saved exploratory pilot to {output}")
    return output
