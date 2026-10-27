import type {
  ConfigStatus,
  DatasetStatistics,
  EvaluationResults,
  MovieDetails,
  RecommendationMode,
  RecommendationResponse,
  StreamEvent,
  UserSearchResult,
  UserSummary,
} from "./types";

// Relative by default: the Vite dev server proxies /api to FastAPI.
const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly status: number,
  ) {
    super(message);
  }
}

const BACKEND_UNAVAILABLE = new ApiError(
  "backend_unavailable",
  "The MovieMatch backend is not reachable. Start it with the backend command in the README, then reload this page.",
  0,
);

async function parseError(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    if (body?.error?.message) return new ApiError(body.error.code, body.error.message, response.status);
  } catch {
    /* not JSON */
  }
  if (response.status === 502 || response.status === 503 || response.status === 504) {
    return BACKEND_UNAVAILABLE;
  }
  return new ApiError("http_error", `The backend returned HTTP ${response.status}.`, response.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(API_BASE + path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw BACKEND_UNAVAILABLE;
  }
  if (!response.ok) throw await parseError(response);
  try {
    return (await response.json()) as T;
  } catch {
    throw BACKEND_UNAVAILABLE;
  }
}

export const api = {
  configStatus: () => request<ConfigStatus>("/api/config/status"),
  statistics: () => request<DatasetStatistics>("/api/dataset/statistics"),
  searchUsers: (query: string, page: number, pageSize = 12) =>
    request<UserSearchResult>(
      `/api/users/search?${new URLSearchParams({ query, page: String(page), pageSize: String(pageSize) })}`,
    ),
  userSummary: (userId: number) => request<UserSummary>(`/api/users/${userId}/summary`),
  movie: (movieId: number) => request<MovieDetails>(`/api/movies/${movieId}`),
  evaluation: () => request<EvaluationResults>("/api/evaluation/results"),
};

export interface RecommendParams {
  mode: RecommendationMode;
  userId: number;
  topN: number;
  userRequest?: string;
  candidateCount?: number;
}

const ENDPOINTS: Record<RecommendationMode, string> = {
  popularity: "/api/recommendations/popularity",
  item_knn: "/api/recommendations/knn",
  als: "/api/recommendations/als",
  als_llm: "/api/recommendations/als-llm",
};

function body(params: RecommendParams): string {
  const payload: Record<string, unknown> = { userId: params.userId, topN: params.topN };
  if (params.mode === "als" || params.mode === "als_llm") payload.candidateCount = params.candidateCount ?? 50;
  if (params.mode === "als_llm") payload.userRequest = params.userRequest ?? "";
  return JSON.stringify(payload);
}

/** Request recommendations, reporting real backend progress events as they arrive. */
export async function recommend(
  params: RecommendParams,
  onEvent?: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<RecommendationResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${ENDPOINTS[params.mode]}?stream=true`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body(params),
      signal,
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw BACKEND_UNAVAILABLE;
  }
  if (!response.ok) throw await parseError(response);
  if (!response.body) throw BACKEND_UNAVAILABLE;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: RecommendationResponse | null = null;

  const handle = (line: string) => {
    if (!line.trim()) return;
    const event = JSON.parse(line) as StreamEvent;
    if (event.type === "error") throw new ApiError(event.error.code, event.error.message, 500);
    if (event.type === "result") result = event.data;
    onEvent?.(event);
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline >= 0) {
      handle(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      newline = buffer.indexOf("\n");
    }
  }
  handle(buffer + decoder.decode());
  if (!result) throw new ApiError("incomplete_response", "The backend closed the connection before sending results.", 0);
  return result;
}
