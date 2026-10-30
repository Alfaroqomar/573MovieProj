from __future__ import annotations

import dataclasses
import json
import shutil

import httpx
import pytest
from fastapi.testclient import TestClient

from moviematch.api.main import create_app
from moviematch.config import LlmSettings
from moviematch.llm.client import LlmError, OpenAICompatibleClient
from tests.conftest import FakeLlm, candidate_ids_from, make_settings, valid_response_for, write_synthetic_movielens


def client_for(settings, llm=None) -> TestClient:
    return TestClient(create_app(settings, llm_client=llm))


def with_llm(settings):
    return dataclasses.replace(
        settings,
        llm=LlmSettings(True, "http://llm.invalid/v1", "secret-key", "test-model", 5, True, 0.0),
    )


def test_status_dataset_unavailable(tmp_path):
    with client_for(make_settings(tmp_path, None)) as client:
        status = client.get("/api/config/status").json()
        assert status["state"] == "dataset_unavailable"
        assert not any(status["modes"].values())
        assert set(status["modes"]) == {
            "popularity", "popularity_weighted", "item_knn", "als", "als_explicit", "stage1", "neumf", "als_llm",
        }
        assert any("MOVIELENS_DATA_PATH" in m["action"] for m in status["messages"])
        assert client.get("/api/dataset/statistics").status_code == 503
        r = client.post("/api/recommendations/als", json={"userId": 1})
        assert r.status_code == 503 and r.json()["error"]["code"] == "dataset_unavailable"


def test_status_not_processed_and_not_trained(tmp_path):
    from moviematch.data.preprocess import preprocess

    data = write_synthetic_movielens(tmp_path / "ml-32m", n_users=40, n_movies=60)
    settings = make_settings(tmp_path, data)
    with client_for(settings) as client:
        assert client.get("/api/config/status").json()["state"] == "not_processed"
    preprocess(settings, log=lambda _: None)
    with client_for(settings) as client:
        status = client.get("/api/config/status").json()
        assert status["state"] == "models_not_trained"
        assert client.get("/api/users/1/summary").status_code == 200
        r = client.post("/api/recommendations/als", json={"userId": 1})
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_ready_without_llm(prepared):
    with client_for(prepared) as client:
        status = client.get("/api/config/status").json()
        assert status["state"] == "ready"
        assert status["modes"]["als_llm"] is False and status["defaultMode"] == "als"
        assert any(m["message"] == "LLM reranking is not configured." for m in status["messages"])
        r = client.post("/api/recommendations/als-llm", json={"userId": 1})
        assert r.status_code == 409 and r.json()["error"]["code"] == "llm_not_configured"


def test_lookups(prepared):
    with client_for(prepared) as client:
        stats = client.get("/api/dataset/statistics").json()
        assert stats["users"] == 300 and stats["movies"] == 400 and stats["mostRatedMovies"]
        search = client.get("/api/users/search", params={"query": "12", "pageSize": 5}).json()
        assert search["users"][0]["userId"] == 12
        assert all(str(u["userId"]).startswith("12") for u in search["users"])
        assert client.get("/api/users/search", params={"query": "abc"}).status_code == 400
        summary = client.get("/api/users/5/summary").json()
        assert summary["ratingCount"] > 0 and summary["highlyRatedMovies"]
        assert all(m["rating"] >= 4.0 for m in summary["highlyRatedMovies"])
        assert client.get("/api/users/999999/summary").status_code == 404
        movie = client.get("/api/movies/1").json()
        assert movie["title"] == "The Synthetic Picture 1" and "mind-bending" in movie["tags"]
        assert client.get("/api/movies/0").status_code == 404
        assert client.get("/api/evaluation/results").json()["available"] in (True, False)


@pytest.mark.parametrize("endpoint,mode", [
    ("popularity", "popularity"), ("knn", "item_knn"), ("als", "als"),
])
def test_classical_recommendations_exclude_rated(prepared, endpoint, mode):
    with client_for(prepared) as client:
        summary = client.get("/api/users/5/summary").json()
        r = client.post(f"/api/recommendations/{endpoint}", json={"userId": 5, "topN": 10})
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == mode and len(data["recommendations"]) == 10
        ranks = [rec["finalRank"] for rec in data["recommendations"]]
        assert ranks == list(range(1, 11))
        top_ids = {m["movieId"] for m in summary["highlyRatedMovies"]}
        assert not top_ids & {rec["movie"]["movieId"] for rec in data["recommendations"]}


def test_als_llm_reranks_only_candidates(prepared):
    llm = FakeLlm([valid_response_for(10)])
    with client_for(with_llm(prepared), llm) as client:
        als = client.post("/api/recommendations/als", json={"userId": 5, "topN": 50}).json()
        als_ids = [r["movie"]["movieId"] for r in als["recommendations"]]
        r = client.post("/api/recommendations/als-llm", json={"userId": 5, "topN": 10, "userRequest": "quiet sci-fi"})
        assert r.status_code == 200
        data = r.json()
    sent = candidate_ids_from(llm.calls[0])
    assert sent == als_ids  # exactly the ALS top-50, in ALS order
    assert "secret-key" not in json.dumps(llm.calls[0])
    assert data["usedFallback"] is False
    recs = data["recommendations"]
    assert [r["movie"]["movieId"] for r in recs] == list(reversed(als_ids))[:10]
    assert recs[0]["originalAlsRank"] == 50 and recs[0]["finalRank"] == 1
    assert all(r["explanationSource"] == "llm" and r["llmRelevanceScore"] is not None for r in recs)
    assert data["interpretedRequest"]["preferredGenres"] == ["Sci-Fi"]


def test_als_llm_falls_back_transparently(prepared):
    def unknown(messages):
        return json.dumps({"interpretedRequest": {"preferredGenres": [], "preferredTags": [], "similarTo": [], "avoid": []},
                           "rankedMovies": [{"movieId": 987654, "rank": 1, "llmRelevanceScore": 1, "explanation": "x"}]})

    llm = FakeLlm([unknown, unknown])
    with client_for(with_llm(prepared), llm) as client:
        data = client.post("/api/recommendations/als-llm", json={"userId": 5, "topN": 5}).json()
        als = client.post("/api/recommendations/als", json={"userId": 5, "topN": 5}).json()
    assert len(llm.calls) == 2
    assert data["usedFallback"] is True
    assert data["fallbackLabel"] == "ALS fallback because LLM reranking failed."
    assert [r["movie"]["movieId"] for r in data["recommendations"]] == [r["movie"]["movieId"] for r in als["recommendations"]]
    assert all(r["usedFallback"] and r["llmRelevanceScore"] is None for r in data["recommendations"])


def test_streaming_progress_events(prepared):
    llm = FakeLlm([LlmError("timeout", "The LLM did not respond within 5 seconds.")])
    with client_for(with_llm(prepared), llm) as client:
        with client.stream("POST", "/api/recommendations/als-llm?stream=true", json={"userId": 5, "topN": 5}) as r:
            assert r.status_code == 200
            events = [json.loads(line) for line in r.iter_lines() if line]
    stages = [(e["stage"], e["status"]) for e in events if e["type"] == "stage"]
    assert stages[:6] == [
        ("loading_history", "started"), ("loading_history", "done"),
        ("generating_als", "started"), ("generating_als", "done"),
        ("finding_similar", "started"), ("finding_similar", "done"),
    ]
    assert ("reranking_llm", "failed") in stages
    result = events[-1]
    assert result["type"] == "result" and result["data"]["usedFallback"] is True
    assert "did not respond" in result["data"]["fallbackReason"]


def test_streaming_invalid_user_is_http_error(prepared):
    with client_for(prepared) as client:
        r = client.post("/api/recommendations/als?stream=true", json={"userId": 999999})
        assert r.status_code == 404 and r.json()["error"]["code"] == "user_not_found"


def test_invalid_request_body(prepared):
    with client_for(prepared) as client:
        r = client.post("/api/recommendations/als", json={"userId": -1})
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_http_client_hides_secrets():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer top-secret"
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(401, text="bad key top-secret")

    settings = LlmSettings(True, "http://llm.invalid/v1/", "top-secret", "m", 5, True, 0.0)
    client = OpenAICompatibleClient(settings, transport=httpx.MockTransport(handler))
    with pytest.raises(LlmError) as exc:
        client.complete([{"role": "user", "content": "hi"}])
    assert "top-secret" not in exc.value.safe_message and "401" in exc.value.safe_message

    def ok(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "m" and body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = OpenAICompatibleClient(settings, transport=httpx.MockTransport(ok))
    assert client.complete([{"role": "user", "content": "hi"}]) == "{}"
    assert "top-secret" not in repr(settings)


def test_retrained_artifacts_need_restart(prepared, tmp_path):
    # Copy processed artifacts without models: status must say models are not trained.
    settings = dataclasses.replace(prepared, artifacts_dir=tmp_path / "artifacts")
    shutil.copytree(prepared.processed_dir, settings.processed_dir)
    with client_for(settings) as client:
        assert client.get("/api/config/status").json()["state"] == "models_not_trained"


@pytest.mark.parametrize("endpoint,mode", [
    ("popularity-weighted", "popularity_weighted"), ("als-explicit", "als_explicit"), ("stage1", "stage1"),
])
def test_new_modes_exclude_rated(prepared, endpoint, mode):
    with client_for(prepared) as client:
        summary = client.get("/api/users/5/summary").json()
        assert summary["splitMode"] == "date" and "isColdStart" in summary
        data = client.post(f"/api/recommendations/{endpoint}", json={"userId": 5, "topN": 10}).json()
    assert data["mode"] == mode and len(data["recommendations"]) == 10
    assert [r["finalRank"] for r in data["recommendations"]] == list(range(1, 11))
    top_ids = {m["movieId"] for m in summary["highlyRatedMovies"]}
    assert not top_ids & {r["movie"]["movieId"] for r in data["recommendations"]}
    if mode == "stage1":
        assert data["candidateCount"] == 50
        assert all(r["candidateSource"] in {"both", "als", "knn", "als_top_up"} for r in data["recommendations"])


def test_neumf_mode(prepared):
    pytest.importorskip("torch")
    with client_for(prepared) as client:
        assert client.get("/api/config/status").json()["modes"]["neumf"] is True
        data = client.post("/api/recommendations/neumf", json={"userId": 5, "topN": 5}).json()
    assert data["mode"] == "neumf" and len(data["recommendations"]) == 5


def test_candidates_export(prepared):
    import pandas as pd

    from moviematch.models.candidates import export_candidates

    summary = export_candidates(prepared, "test", n_users=20, log=lambda _: None)
    frame = pd.read_parquet(summary["candidates"])
    assert {"user_id", "rank", "movie_id", "merged_score", "source", "knn_evidence", "is_cold_start"} <= set(frame.columns)
    assert frame.groupby("user_id")["rank"].max().max() == 50
    lines = [json.loads(line) for line in open(summary["userFile"])]
    assert len(lines) == summary["users"] and all(len(u["recent_5"]) <= 5 for u in lines)
