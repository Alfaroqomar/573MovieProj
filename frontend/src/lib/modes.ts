import type { RecommendationMode, StageName } from "../api/types";

export const MODES: Array<{
  id: RecommendationMode;
  label: string;
  short: string;
  description: string;
  colorVar: string;
}> = [
  {
    id: "popularity",
    label: "Popularity",
    short: "Popularity",
    description: "The most-rated movies you have not rated yet. The same list for everyone (proposal baseline).",
    colorVar: "var(--color-model-popularity)",
  },
  {
    id: "popularity_weighted",
    label: "Popularity (weighted)",
    short: "Pop. weighted",
    description: "Popular movies ranked by a Bayesian weighted average of their ratings, so well-rated movies rise.",
    colorVar: "var(--color-model-popularity-weighted)",
  },
  {
    id: "item_knn",
    label: "Item-KNN",
    short: "Item-KNN",
    description: "Movies whose rating patterns are most similar to the movies you rated highly.",
    colorVar: "var(--color-model-knn)",
  },
  {
    id: "als",
    label: "ALS",
    short: "ALS",
    description: "Matrix factorization learns your taste from all your ratings and scores every unseen movie.",
    colorVar: "var(--color-model-als)",
  },
  {
    id: "als_explicit",
    label: "ALS (explicit ratings)",
    short: "ALS explicit",
    description: "The proposal's ALS: predicts the star rating you would give, then ranks by that prediction.",
    colorVar: "var(--color-model-als-explicit)",
  },
  {
    id: "neumf",
    label: "NeuMF",
    short: "NeuMF",
    description: "Neural collaborative filtering, the deep-learning baseline. Makes its own list, separate from the pipeline.",
    colorVar: "var(--color-model-neumf)",
  },
  {
    id: "stage1",
    label: "Stage 1 (ALS + KNN merged)",
    short: "Stage 1",
    description: "The proposal's shortlist: ALS's top 30 and Item-KNN's top 30, merged into 50. Shows the top of that list.",
    colorVar: "var(--color-model-stage1)",
  },
  {
    id: "als_llm",
    label: "Stage 1 + LLM reranking",
    short: "Stage 1 + LLM",
    description: "The LLM reorders a 50-movie shortlist from ALS and Item-KNN.",
    colorVar: "var(--color-model-llm)",
  },
];

export const modeInfo = (mode: RecommendationMode) => MODES.find((m) => m.id === mode)!;

export const STAGE_LABELS: Record<StageName, string> = {
  loading_history: "Loading user history",
  ranking_popularity: "Ranking popular movies",
  generating_als: "Generating ALS candidates",
  finding_similar: "Finding similar movies",
  merging_candidates: "Merging ALS and Item-KNN shortlists",
  scoring_neumf: "Scoring every movie with NeuMF",
  reranking_llm: "Reranking with the LLM",
  validating: "Validating the response",
};

export const STAGES_BY_MODE: Record<RecommendationMode, StageName[]> = {
  popularity: ["loading_history", "ranking_popularity"],
  popularity_weighted: ["loading_history", "ranking_popularity"],
  item_knn: ["loading_history", "finding_similar"],
  als: ["loading_history", "generating_als", "finding_similar"],
  als_explicit: ["loading_history", "generating_als", "finding_similar"],
  stage1: ["loading_history", "generating_als", "finding_similar", "merging_candidates"],
  neumf: ["loading_history", "scoring_neumf"],
  als_llm: ["loading_history", "generating_als", "finding_similar", "merging_candidates", "reranking_llm", "validating"],
};

const integer = new Intl.NumberFormat("en-US");
export const fmtInt = (n: number) => integer.format(n);

export function fmtCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 1)}M`;
  if (n >= 10_000) return `${Math.round(n / 1000)}K`;
  return integer.format(n);
}

export const fmtScore = (n: number | null | undefined, digits = 2) =>
  n === null || n === undefined ? "n/a" : n.toFixed(digits);

export const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

export function fmtDate(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleDateString("en-US", { year: "numeric", month: "short", timeZone: "UTC" });
}
