"""Strict schema and validation for the LLM reranking response."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, ValidationError

MAX_EXPLANATION_CHARS = 300
MAX_LIST_ITEMS = 10
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(?P<body>.*)\n```$", re.DOTALL)


class InterpretedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferredGenres: list[StrictStr] = Field(max_length=MAX_LIST_ITEMS)
    preferredTags: list[StrictStr] = Field(max_length=MAX_LIST_ITEMS)
    similarTo: list[StrictStr] = Field(max_length=MAX_LIST_ITEMS)
    avoid: list[StrictStr] = Field(max_length=MAX_LIST_ITEMS)


class RankedMovie(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movieId: StrictInt
    rank: StrictInt
    llmRelevanceScore: StrictFloat
    explanation: StrictStr


class LlmRerankOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interpretedRequest: InterpretedRequest
    rankedMovies: list[RankedMovie] = Field(min_length=1)


@dataclass
class ValidationOutcome:
    output: LlmRerankOutput | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.output is not None and not self.errors


def _reject_constant(name: str) -> float:
    raise ValueError(f"non-standard JSON constant {name}")


def _strip_single_fence(text: str) -> str:
    """Allow exactly one enclosing ```json fence; anything else must be bare JSON."""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group("body").strip() if match else stripped


def _clean_short_list(values: list[str]) -> list[str]:
    seen, out = set(), []
    for value in values:
        cleaned = " ".join(value.split())[:80]
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out


def validate_llm_output(
    text: str,
    candidate_ids: list[int],
    expected_count: int,
    genre_vocabulary: list[str],
    max_explanation_chars: int = MAX_EXPLANATION_CHARS,
) -> ValidationOutcome:
    # 1. Strict JSON.
    try:
        data = json.loads(_strip_single_fence(text), parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        return ValidationOutcome(None, [f"Response is not valid JSON ({exc})."])
    if not isinstance(data, dict):
        return ValidationOutcome(None, ["Response must be a single JSON object."])

    # 2 & 5. Pydantic schema, including required fields and forbidden extras.
    try:
        parsed = LlmRerankOutput.model_validate(data)
    except ValidationError as exc:
        errors = []
        for err in exc.errors()[:10]:
            location = ".".join(str(part) for part in err["loc"]) or "(root)"
            errors.append(f"{location}: {err['msg']}")
        return ValidationOutcome(None, errors)

    errors: list[str] = []
    warnings: list[str] = []
    allowed = set(candidate_ids)
    movies = parsed.rankedMovies

    # 3 & 9. Every movie must come from the supplied candidate pool.
    unknown = [m.movieId for m in movies if m.movieId not in allowed]
    if unknown:
        errors.append(
            "movieId values not in the candidate list: " + ", ".join(map(str, unknown[:10]))
        )

    # 4. No duplicates.
    ids = [m.movieId for m in movies]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        errors.append("Duplicate movieId values: " + ", ".join(map(str, duplicates)))

    # Exact number of results requested.
    if len(movies) != expected_count:
        errors.append(f"Expected exactly {expected_count} rankedMovies, received {len(movies)}.")

    # 7. Ranks unique and sequential from 1.
    ranks = sorted(m.rank for m in movies)
    if ranks != list(range(1, len(movies) + 1)):
        errors.append("rank values must be unique and sequential starting at 1.")

    # 6. Scores: reject non-finite values, clamp to [0, 1].
    for movie in movies:
        score = movie.llmRelevanceScore
        if not math.isfinite(score):
            errors.append(f"llmRelevanceScore for movieId {movie.movieId} is not a finite number.")
        elif score < 0.0 or score > 1.0:
            movie.llmRelevanceScore = min(1.0, max(0.0, score))
            warnings.append(f"Clamped llmRelevanceScore for movieId {movie.movieId} to [0, 1].")

    # 8. Concise, non-empty explanations.
    for movie in movies:
        movie.explanation = " ".join(movie.explanation.split())
        if not movie.explanation:
            errors.append(f"Explanation for movieId {movie.movieId} is empty.")
        elif len(movie.explanation) > max_explanation_chars:
            errors.append(
                f"Explanation for movieId {movie.movieId} exceeds {max_explanation_chars} characters."
            )

    if errors:
        return ValidationOutcome(None, errors, warnings)

    # Interpreted request: keep only MovieLens genre names; tidy the rest.
    vocabulary = {g.lower(): g for g in genre_vocabulary}
    interpreted = parsed.interpretedRequest
    kept = [vocabulary[g.lower()] for g in interpreted.preferredGenres if g.lower() in vocabulary]
    dropped = [g for g in interpreted.preferredGenres if g.lower() not in vocabulary]
    if dropped:
        warnings.append("Dropped non-MovieLens genre names from interpretedRequest: " + ", ".join(dropped))
    interpreted.preferredGenres = _clean_short_list(kept)
    interpreted.preferredTags = _clean_short_list(interpreted.preferredTags)
    interpreted.similarTo = _clean_short_list(interpreted.similarTo)
    interpreted.avoid = _clean_short_list(interpreted.avoid)

    parsed.rankedMovies = sorted(movies, key=lambda m: m.rank)
    return ValidationOutcome(parsed, [], warnings)
