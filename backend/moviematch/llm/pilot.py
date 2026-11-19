"""Small, reproducible LLM comparison on the validation period."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from moviematch.config import Settings
from moviematch.data.paths import model_paths, processed_paths
from moviematch.llm.client import OpenAICompatibleClient
from moviematch.llm.reranker import rerank
from moviematch.models.candidates import export_candidates
from moviematch.models.evaluation import ranking_metrics


def run_validation_pilot(settings: Settings, users: int = 5, log=print, *, split: str = "validation") -> Path:
    if not settings.llm.configured:
        raise RuntimeError("Configure an LLM before running the pilot.")
    if settings.split_mode != "date":
        raise RuntimeError("The validation pilot requires SPLIT_MODE=date.")
    if users < 1:
        raise ValueError("users must be at least 1")

    expected_fit = {"validation": "train", "test": "train+validation"}[split]
    training = json.loads(model_paths(settings).training_manifest.read_text())["models"]
    if any(training.get(name, {}).get("fitOn") != expected_fit for name in ("als", "knn")):
        raise RuntimeError(f"The {split} run requires ALS and Item-KNN fitted on {expected_fit}.")
    export = export_candidates(settings, split_name=split, n_users=users, log=log)
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
            als = profile["als_top10"]
            knn = profile["knn_top10"]
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
                        "tags": list(row.tags)[:3],
                        "mergedScore": float(row.merged_score),
                        "stage1Rank": int(row.rank),
                        "alsScore": None if pd.isna(row.als_score) else float(row.als_score),
                        "knnScore": None if pd.isna(row.knn_score) else float(row.knn_score),
                        "knnEvidence": json.loads(row.knn_evidence)[:1],
                    }
                    for row in candidates.itertuples()
                ],
            }
            started = time.perf_counter()
            result = rerank(client, payload, 10, genres)
            elapsed = time.perf_counter() - started
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
                        "knnEvidence": by_id[movie.movieId]["knnEvidence"],
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
                "llmSeconds": round(elapsed, 2),
                "fallbackReason": result.fallback_reason,
                "validationErrors": result.validation_errors,
                "metrics": user_metrics,
                "top10": {"stage1": stage1[:10], "als": als, "itemKnn": knn, "llmOrFallback": llm},
                "explanations": explanations,
            })
            log(f"  {index}/{len(profiles)} user {user_id}: {'LLM' if result.succeeded else 'fallback'}")
    finally:
        client.close()

    if not records:
        raise RuntimeError("There are no eligible users to evaluate.")
    differences = np.subtract(metrics["llm_or_fallback"]["ndcg"], metrics["stage1"]["ndcg"])
    samples = np.random.default_rng(settings.eval_random_seed).choice(
        differences, size=(2000, len(differences)), replace=True,
    )
    report = {
        "kind": "sampled_llm_evaluation",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "sampleSeed": settings.eval_random_seed,
        "trainedOn": expected_fit,
        "model": settings.llm.model,
        "users": len(records),
        "coldStartUsers": sum(record["coldStart"] for record in records),
        "llmSuccesses": sum(not record["usedFallback"] for record in records),
        "llmFallbacks": sum(record["usedFallback"] for record in records),
        "medianLlmSeconds": round(float(np.median([record["llmSeconds"] for record in records])), 2),
        "pairedNdcgAt10VsStage1": {
            "meanDifference": round(float(differences.mean()), 5),
            "bootstrap95": np.round(np.quantile(samples.mean(axis=1), [0.025, 0.975]), 5).tolist(),
        },
        "metricMeans": {
            name: {metric + "@10": round(float(np.mean(values)), 5) for metric, values in scores.items()}
            for name, scores in metrics.items()
        },
        "notes": [
            "All methods use the same users and held-out ratings.",
            "The LLM receives only user ratings and tags available before the evaluated period.",
            "The LLM-or-fallback metrics include Stage 1 fallback when an LLM response is invalid.",
            "Explanations are generated by the app from supplied evidence.",
        ],
        "records": records,
    }
    output = settings.artifacts_dir / "llm_pilot" / f"{split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    log(f"Saved exploratory pilot to {output}")
    return output
