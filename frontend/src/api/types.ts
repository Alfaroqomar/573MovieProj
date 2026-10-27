// Shapes returned by the FastAPI backend (camelCase JSON).

export type RecommendationMode = "popularity" | "item_knn" | "als" | "als_llm";

export interface Movie {
  movieId: number;
  title: string;
  year?: number | null;
  genres: string[];
  tags: string[];
  posterUrl?: string | null;
}

export interface MovieDetails extends Movie {
  ratingCount: number;
  meanRating: number | null;
  imdbId: string | null;
  tmdbId: number | null;
}

export interface HighlyRatedMovie {
  movieId: number;
  title: string;
  year?: number | null;
  rating: number;
}

export interface UserSummary {
  userId: number;
  ratingCount: number;
  trainingRatingCount: number;
  meanRating: number;
  firstRatingAt: string;
  lastRatingAt: string;
  favoriteGenres: string[];
  highlyRatedMovies: HighlyRatedMovie[];
  highRatingThreshold: number;
}

export interface UserSearchResult {
  query: string;
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
  users: Array<{ userId: number; ratingCount: number; favoriteGenres: string[] }>;
}

export interface KnnEvidence {
  historyMovieId: number;
  historyMovieTitle: string;
  similarity: number;
  historyRating: number;
}

export interface Recommendation {
  movie: Movie;
  mode: RecommendationMode;
  finalRank: number;
  originalAlsRank?: number | null;
  alsScore?: number | null;
  alsRawScore?: number | null;
  knnScore?: number | null;
  strongestKnnSimilarity?: number | null;
  knnEvidence?: KnnEvidence[] | null;
  llmRelevanceScore?: number | null;
  popularityScore?: number | null;
  ratingCount?: number | null;
  meanRating?: number | null;
  explanation?: string | null;
  explanationSource: "llm" | "template";
  usedFallback: boolean;
}

export interface InterpretedRequest {
  preferredGenres: string[];
  preferredTags: string[];
  similarTo: string[];
  avoid: string[];
}

export interface RecommendationResponse {
  mode: RecommendationMode;
  userId: number;
  topN: number;
  candidateCount?: number | null;
  recommendations: Recommendation[];
  usedFallback: boolean;
  fallbackLabel?: string | null;
  fallbackReason?: string | null;
  interpretedRequest?: InterpretedRequest | null;
  llm?: { attempts: number; model: string } | null;
  stageTimingsMs: Record<string, number>;
  elapsedMs: number;
  message?: string | null;
}

export interface StatusMessage {
  level: "error" | "warning" | "info";
  code: string;
  message: string;
  action: string;
}

export type ReadinessState =
  | "dataset_unavailable"
  | "not_processed"
  | "models_not_trained"
  | "partial"
  | "ready";

export interface ConfigStatus {
  state: ReadinessState;
  rawDataset: { configured: boolean; found: boolean; problem: string | null };
  processed: boolean;
  processedProblem: string | null;
  models: Record<RecommendationMode, boolean>;
  modes: Record<RecommendationMode, boolean>;
  defaultMode: RecommendationMode;
  llm: {
    enabled: boolean;
    configured: boolean;
    model: string | null;
    baseUrlSet: boolean;
    apiKeySet: boolean;
    timeoutSeconds: number;
    jsonMode: boolean;
    missing: string[];
  };
  evaluationAvailable: boolean;
  posterSourceConfigured: boolean;
  messages: StatusMessage[];
}

export interface DatasetStatistics {
  dataset: string;
  ratings: number;
  users: number;
  movies: number;
  ratedMovies: number;
  tagApplications: number;
  uniqueTags: number;
  links: number;
  meanRating: number;
  ratingDistribution: Record<string, number>;
  firstRatingAt: string;
  lastRatingAt: string;
  medianRatingsPerUser: number;
  split: { train: number; validation: number; test: number };
  genres: Record<string, number>;
  mostRatedMovies: Array<Movie & { ratingCount: number }>;
  processedAt: string;
  matchesPublishedMl32mCounts: boolean;
}

export interface EvaluationResults {
  available: boolean;
  message?: string;
  split?: string;
  k?: number;
  usersEvaluated?: number;
  eligibleUsers?: number;
  relevanceThreshold?: number;
  protocol?: string;
  models?: Record<string, Record<string, number>>;
  notEvaluated?: Record<string, string>;
  createdAt?: string;
}

export type StageName =
  | "loading_history"
  | "ranking_popularity"
  | "generating_als"
  | "finding_similar"
  | "reranking_llm"
  | "validating";

export type StreamEvent =
  | { type: "stage"; stage: StageName; status: "started" | "done" | "failed"; attempt?: number }
  | { type: "result"; data: RecommendationResponse }
  | { type: "error"; error: { code: string; message: string } };
