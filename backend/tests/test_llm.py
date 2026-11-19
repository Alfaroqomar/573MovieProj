from __future__ import annotations

import json

from moviematch.llm.client import LlmError
from moviematch.llm.reranker import rerank
from moviematch.llm.schemas import validate_llm_output
from tests.conftest import FakeLlm

GENRES = ["Action", "Drama", "Sci-Fi"]
CANDIDATES = [11, 22, 33, 44]


def body(movies, interpreted=None, **extra):
    return json.dumps({
        "interpretedRequest": interpreted or {"preferredGenres": [], "preferredTags": [], "similarTo": [], "avoid": []},
        "rankedMovies": movies,
        **extra,
    })


def movie(mid, rank, score=0.5, explanation="Fits the supplied genres."):
    return {"movieId": mid, "rank": rank, "llmRelevanceScore": score, "explanation": explanation}


def check(text, expected=2):
    return validate_llm_output(text, CANDIDATES, expected, GENRES)


def test_accepts_valid_response_and_sorts_by_rank():
    outcome = check(body([movie(33, 2), movie(11, 1)]))
    assert outcome.ok
    assert [m.movieId for m in outcome.output.rankedMovies] == [11, 33]


def test_rejects_non_json_and_prose():
    assert not check("Here are my picks: 11, 33").ok
    assert not check("```json\n" + body([movie(11, 1), movie(22, 2)]) + "\n``` thanks").ok


def test_allows_single_enclosing_fence():
    assert check("```json\n" + body([movie(11, 1), movie(22, 2)]) + "\n```").ok


def test_rejects_unknown_movie():
    outcome = check(body([movie(11, 1), movie(999, 2)]))
    assert not outcome.ok and any("not in the candidate list" in e for e in outcome.errors)


def test_rejects_duplicates_rank_gaps_and_wrong_count():
    assert not check(body([movie(11, 1), movie(11, 2)])).ok
    assert not check(body([movie(11, 1), movie(22, 3)])).ok
    assert not check(body([movie(11, 1)])).ok


def test_rejects_missing_and_extra_fields():
    assert not check(json.dumps({"rankedMovies": [movie(11, 1), movie(22, 2)]})).ok
    bad = movie(11, 1)
    del bad["explanation"]
    assert not check(body([bad, movie(22, 2)])).ok
    assert not check(body([movie(11, 1), movie(22, 2)], extra="x")).ok
    assert not check(body([{**movie(11, 1), "movieId": "11"}, movie(22, 2)])).ok


def test_scores_are_clamped_or_rejected():
    outcome = check(body([movie(11, 1, score=1.7), movie(22, 2, score=-3)]))
    assert outcome.ok
    assert [m.llmRelevanceScore for m in outcome.output.rankedMovies] == [1.0, 0.0]
    assert not check(body([movie(11, 1), movie(22, 2)]).replace("0.5", "NaN", 1)).ok


def test_rejects_long_or_empty_explanations():
    assert not check(body([movie(11, 1, explanation="x" * 400), movie(22, 2)])).ok
    assert not check(body([movie(11, 1, explanation="   "), movie(22, 2)])).ok


def test_interpreted_genres_limited_to_movielens_vocabulary():
    interpreted = {"preferredGenres": ["sci-fi", "Science Fiction"], "preferredTags": ["thoughtful"], "similarTo": [], "avoid": []}
    outcome = check(body([movie(11, 1), movie(22, 2)], interpreted=interpreted))
    assert outcome.ok and outcome.output.interpretedRequest.preferredGenres == ["Sci-Fi"]


PAYLOAD = {"userRequest": "x", "userProfile": {}, "candidates": [{"movieId": i} for i in CANDIDATES]}


def test_compact_ranker_rejects_duplicates_and_uses_supplied_explanation():
    def compact(*ids):
        return body([{"movieId": movie_id, "llmRelevanceScore": 0.7} for movie_id in ids])

    payload = {
        **PAYLOAD,
        "userProfile": {"preferredGenres": ["Drama"]},
        "candidates": [{"movieId": 11, "genres": ["Drama"]}, {"movieId": 22, "genres": ["Action"]}],
    }
    llm = FakeLlm([compact(22, 22), compact(22, 11)])
    result = rerank(llm, payload, 2, GENRES)
    assert result.attempts == 2 and result.succeeded
    assert [(m.movieId, m.rank) for m in result.output.rankedMovies] == [(22, 1), (11, 2)]
    assert result.output.rankedMovies[1].explanation == "Matches genres in your rating profile: Drama."


def test_rerank_retries_once_then_succeeds():
    good = body([movie(44, 1), movie(11, 2)])
    llm = FakeLlm(["not json", good])
    result = rerank(llm, PAYLOAD, 2, GENRES)
    assert result.succeeded and result.attempts == 2
    assert "rejected" in llm.calls[1][0]["content"]
    schema = llm.response_formats[0]["json_schema"]["schema"]["properties"]["rankedMovies"]
    assert schema["minItems"] == schema["maxItems"] == 2
    assert schema["items"]["properties"]["movieId"]["enum"] == CANDIDATES


def test_rerank_falls_back_after_two_failures():
    llm = FakeLlm(["not json", body([movie(999, 1), movie(11, 2)])])
    result = rerank(llm, PAYLOAD, 2, GENRES)
    assert not result.succeeded and result.attempts == 2
    assert "failed validation twice" in result.fallback_reason


def test_rerank_does_not_retry_timeouts():
    llm = FakeLlm([LlmError("timeout", "The LLM did not respond within 5 seconds.")])
    result = rerank(llm, PAYLOAD, 2, GENRES)
    assert not result.succeeded and result.attempts == 1 and len(llm.calls) == 1
    assert "did not respond" in result.fallback_reason
