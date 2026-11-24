"""Pydantic request and response models for the HTTP API (camelCase JSON)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Mode = Literal[
    "popularity",
    "popularity_weighted",
    "item_knn",
    "als",
    "als_explicit",
    "stage1",
    "neumf",
    "als_llm",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecommendationRequest(_Model):
    userId: int = Field(gt=0, description="MovieLens user ID")
    topN: int = Field(default=10, ge=1, le=50)


class AlsRecommendationRequest(RecommendationRequest):
    candidateCount: int = Field(default=50, ge=20, le=100)


class LlmRecommendationRequest(AlsRecommendationRequest):
    userRequest: str = Field(default="", max_length=500)


class MovieOut(BaseModel):
    movieId: int
    title: str
    year: int | None = None
    genres: list[str]
    tags: list[str]
    posterUrl: str | None = None


class KnnEvidenceOut(BaseModel):
    historyMovieId: int
    historyMovieTitle: str
    similarity: float
    historyRating: float


class RecommendationOut(BaseModel):
    movie: MovieOut
    mode: Mode
    finalRank: int
    stage1Rank: int | None = None
    originalAlsRank: int | None = None
    alsScore: float | None = None
    alsRawScore: float | None = None
    knnScore: float | None = None
    strongestKnnSimilarity: float | None = None
    knnEvidence: list[KnnEvidenceOut] | None = None
    llmRelevanceScore: float | None = None
    predictedRating: float | None = None
    mergedScore: float | None = None
    knnRank: int | None = None
    candidateSource: Literal["both", "als", "knn", "als_top_up"] | None = None
    neumfScore: float | None = None
    popularityScore: float | None = None
    ratingCount: int | None = None
    meanRating: float | None = None
    explanation: str | None = None
    explanationSource: Literal["llm", "template"]
    usedFallback: bool


class InterpretedRequestOut(BaseModel):
    preferredGenres: list[str]
    preferredTags: list[str]
    similarTo: list[str]
    avoid: list[str]


class LlmInfo(BaseModel):
    attempts: int
    model: str


class RecommendationResponse(BaseModel):
    mode: Mode
    userId: int
    topN: int
    candidateCount: int | None = None
    candidateSources: dict[str, int] | None = None
    knnScoring: str | None = None
    recommendations: list[RecommendationOut]
    usedFallback: bool
    fallbackLabel: str | None = None
    fallbackReason: str | None = None
    interpretedRequest: InterpretedRequestOut | None = None
    requestFilter: dict[str, Any] | None = None
    llm: LlmInfo | None = None
    stageTimingsMs: dict[str, float]
    elapsedMs: float
    message: str | None = None
