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
    description: "Widely rated, well-rated movies you have not rated yet, ranked by a Bayesian weighted average.",
    colorVar: "var(--color-model-popularity)",
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
    id: "als_llm",
    label: "ALS + LLM reranking",
    short: "ALS + LLM",
    description: "ALS picks 50 candidates; the LLM reorders only those to match what you ask for and explains each pick.",
    colorVar: "var(--color-model-llm)",
  },
];

export const modeInfo = (mode: RecommendationMode) => MODES.find((m) => m.id === mode)!;

export const STAGE_LABELS: Record<StageName, string> = {
  loading_history: "Loading user history",
  ranking_popularity: "Ranking popular movies",
  generating_als: "Generating ALS candidates",
  finding_similar: "Finding similar movies",
  reranking_llm: "Reranking with the LLM",
  validating: "Validating the response",
};

export const STAGES_BY_MODE: Record<RecommendationMode, StageName[]> = {
  popularity: ["loading_history", "ranking_popularity"],
  item_knn: ["loading_history", "finding_similar"],
  als: ["loading_history", "generating_als", "finding_similar"],
  als_llm: ["loading_history", "generating_als", "finding_similar", "reranking_llm", "validating"],
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
