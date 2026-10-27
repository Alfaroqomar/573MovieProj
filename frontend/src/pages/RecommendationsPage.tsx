import { useEffect, useId, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ApiError, recommend } from "../api/client";
import type { Recommendation, RecommendationMode, RecommendationResponse, StageName } from "../api/types";
import { Alert } from "../components/Alert";
import { MovieCard } from "../components/MovieCard";
import { MovieDetailsDialog } from "../components/MovieDetailsDialog";
import { ProgressSteps, type StageState } from "../components/ProgressSteps";
import { UserPicker, UserSummaryCard, useUserSummary } from "../components/UserPicker";
import { Button, Panel } from "../components/ui";
import { MODES, fmtMs, modeInfo } from "../lib/modes";
import { datasetReady, useStatus } from "../lib/status";

const TOP_N_OPTIONS = [5, 10, 20];

function modeUnavailableReason(mode: RecommendationMode, available: boolean, llmConfigured: boolean): string | null {
  if (available) return null;
  if (mode === "als_llm" && !llmConfigured) return "LLM reranking is not configured.";
  return "Model not trained or not loaded.";
}

export function RecommendationsPage() {
  const { status } = useStatus();
  const [params, setParams] = useSearchParams();
  const initialUser = Number(params.get("user"));
  const [userId, setUserId] = useState<number | null>(Number.isInteger(initialUser) && initialUser > 0 ? initialUser : null);
  const [mode, setMode] = useState<RecommendationMode | null>(null);
  const [topN, setTopN] = useState(10);
  const [userRequest, setUserRequest] = useState("");
  const [running, setRunning] = useState(false);
  const [stages, setStages] = useState<Partial<Record<StageName, StageState>>>({});
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [details, setDetails] = useState<Recommendation | null>(null);
  const abort = useRef<AbortController | null>(null);
  const requestId = useId();
  const resultsRef = useRef<HTMLHeadingElement>(null);

  const { summary, error: summaryError, loading: summaryLoading } = useUserSummary(userId);
  const ready = datasetReady(status);

  // Default mode: ALS + LLM when configured, otherwise ALS.
  useEffect(() => {
    if (status && mode === null) setMode(status.defaultMode);
  }, [status, mode]);

  useEffect(() => () => abort.current?.abort(), []);

  const selectUser = (id: number) => {
    setUserId(id);
    setResult(null);
    setError(null);
    setParams({ user: String(id) }, { replace: true });
  };

  const activeMode = mode ?? "als";
  const modeAvailable = status?.modes?.[activeMode] ?? false;

  const generate = async () => {
    if (!userId || !summary) {
      setError(new ApiError("invalid_user", "Select a valid MovieLens user first.", 400));
      return;
    }
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setRunning(true);
    setError(null);
    setResult(null);
    setStages({});
    try {
      const response = await recommend(
        { mode: activeMode, userId, topN, userRequest: activeMode === "als_llm" ? userRequest : undefined },
        (event) => {
          if (event.type !== "stage") return;
          setStages((prev) => ({
            ...prev,
            [event.stage]: event.status === "started" ? "active" : event.status === "done" ? "done" : "failed",
          }));
        },
        controller.signal,
      );
      setResult(response);
      requestAnimationFrame(() => resultsRef.current?.focus());
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(e as ApiError);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="grid gap-8 lg:grid-cols-[21rem_1fr] lg:items-start">
      <h1 className="sr-only">Recommendations</h1>
      <Panel className="space-y-6 p-5 lg:sticky lg:top-4">
        <UserPicker selectedUserId={userId} onSelect={selectUser} disabled={!ready} />

        {userId && summaryLoading ? <p className="text-sm text-muted" role="status">Loading user {userId}…</p> : null}
        {summaryError ? (
          <Alert tone="error" title={summaryError.code === "user_not_found" ? "User not found" : "Could not load user"} live>
            {summaryError.message}
          </Alert>
        ) : null}
        {summary ? <UserSummaryCard summary={summary} compact /> : null}

        <fieldset>
          <legend className="mb-2 text-sm font-semibold text-paper">Recommendation mode</legend>
          <div className="space-y-1.5">
            {MODES.map((m) => {
              const available = status?.modes?.[m.id] ?? false;
              const reason = modeUnavailableReason(m.id, available, !!status?.llm.configured);
              const checked = activeMode === m.id;
              return (
                <label
                  key={m.id}
                  className={`flex cursor-pointer gap-3 rounded-[var(--radius-control)] border p-3 ${
                    checked ? "border-rose/70 bg-raised" : "border-line hover:border-muted"
                  } ${!available ? "cursor-not-allowed opacity-70" : ""}`}
                >
                  <input
                    type="radio"
                    name="mode"
                    value={m.id}
                    checked={checked}
                    disabled={!available}
                    onChange={() => setMode(m.id)}
                    className="mt-1 h-4 w-4 accent-[var(--color-wine)]"
                  />
                  <span className="min-w-0">
                    <span className="flex items-center gap-2 text-sm font-semibold text-paper">
                      <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: m.colorVar }} />
                      {m.label}
                    </span>
                    <span className="mt-0.5 block text-xs text-muted">{reason ?? m.description}</span>
                  </span>
                </label>
              );
            })}
          </div>
        </fieldset>

        <fieldset>
          <legend className="mb-2 text-sm font-semibold text-paper">Number of recommendations</legend>
          <div className="grid grid-cols-3 gap-1.5">
            {TOP_N_OPTIONS.map((n) => (
              <label
                key={n}
                className={`flex min-h-11 cursor-pointer items-center justify-center rounded-[var(--radius-control)] border text-sm font-semibold has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-rose ${
                  topN === n ? "border-rose/70 bg-raised text-paper" : "border-line text-muted hover:text-paper"
                }`}
              >
                <input type="radio" name="topN" value={n} checked={topN === n} onChange={() => setTopN(n)} className="sr-only" />
                Top {n}
              </label>
            ))}
          </div>
          {activeMode === "als_llm" ? (
            <p className="mt-1.5 text-xs text-muted">ALS generates 50 candidates; the LLM returns the top {topN}.</p>
          ) : null}
        </fieldset>

        {activeMode === "als_llm" ? (
          <div>
            <label htmlFor={requestId} className="mb-1.5 block text-sm font-semibold text-paper">
              What are you in the mood for?
            </label>
            <textarea
              id={requestId}
              rows={3}
              maxLength={500}
              value={userRequest}
              onChange={(e) => setUserRequest(e.target.value)}
              placeholder="I want a thoughtful science-fiction movie similar to Interstellar but with less action."
              aria-describedby={`${requestId}-help`}
              className="w-full rounded-[var(--radius-control)] border border-line bg-raised px-3 py-2 text-sm text-paper placeholder:text-faint"
            />
            <p id={`${requestId}-help`} className="mt-1 text-xs text-muted">
              The LLM will only reorder movies already selected by ALS.
            </p>
          </div>
        ) : null}

        {status && !status.llm.configured ? (
          <p className="text-xs text-muted">
            LLM reranking is not configured.
            {status.models.als ? " Popularity, Item-KNN, and ALS remain available." : ""}
          </p>
        ) : null}

        <Button className="w-full" onClick={generate} disabled={running || !ready || !modeAvailable || !summary}>
          {running ? "Generating…" : "Generate Recommendations"}
        </Button>
        {!summary && ready ? <p className="-mt-4 text-xs text-muted">Select a user to enable this button.</p> : null}
      </Panel>

      <section aria-labelledby="results-title" className="min-w-0 space-y-5">
        <h2 id="results-title" ref={resultsRef} tabIndex={-1} className="display-soft text-2xl text-paper outline-none">
          {result
            ? `Top ${result.recommendations.length} for user ${result.userId}`
            : "Recommendations"}
        </h2>

        {running || (Object.keys(stages).length > 0 && !result) ? <ProgressSteps mode={activeMode} states={stages} /> : null}

        {error ? (
          <Alert tone="error" title={errorTitle(error)} live>
            {error.message}
          </Alert>
        ) : null}

        {result ? <ResultHeader result={result} /> : null}

        {result && result.recommendations.length === 0 ? (
          <Alert tone="info" title="No recommendations">
            {result.message ?? "The model returned no movies for this user."}
          </Alert>
        ) : null}

        {result ? (
          <ul className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {result.recommendations.map((rec) => (
              <li key={rec.movie.movieId} className="flex">
                <MovieCard rec={rec} onDetails={setDetails} />
              </li>
            ))}
          </ul>
        ) : null}

        {!result && !running && !error && Object.keys(stages).length === 0 ? (
          <p className="measure text-muted">
            {ready
              ? "Choose a user and a mode, then generate recommendations. Results come from the trained models on this server."
              : "Recommendations become available once the dataset is processed and the models are trained."}
          </p>
        ) : null}
      </section>

      <MovieDetailsDialog rec={details} onClose={() => setDetails(null)} />
    </div>
  );
}

function errorTitle(error: ApiError): string {
  switch (error.code) {
    case "user_not_found":
    case "invalid_user":
      return "Invalid user";
    case "model_unavailable":
      return "Model unavailable";
    case "llm_not_configured":
      return "LLM reranking is not configured";
    case "backend_unavailable":
      return "Backend unavailable";
    case "dataset_unavailable":
    case "not_processed":
    case "models_not_trained":
      return "Dataset not ready";
    default:
      return "Could not generate recommendations";
  }
}

function ResultHeader({ result }: { result: RecommendationResponse }) {
  const info = modeInfo(result.mode);
  const ir = result.interpretedRequest;
  return (
    <div className="space-y-3">
      <p className="num text-sm text-muted">
        {info.label}
        {result.candidateCount ? `, ${result.candidateCount} ALS candidates` : ""}, answered in {fmtMs(result.elapsedMs)}
        {result.llm ? `, ${result.llm.attempts} LLM ${result.llm.attempts === 1 ? "call" : "calls"}` : ""}
      </p>
      {result.usedFallback ? (
        <Alert tone="warning" title={result.fallbackLabel ?? "ALS fallback because LLM reranking failed."} live>
          {result.fallbackReason} The list below is the original ALS order.
        </Alert>
      ) : null}
      {ir ? (
        <div className="rounded-[var(--radius-control)] border border-line bg-panel px-4 py-3 text-sm">
          <p className="font-semibold text-paper">How the LLM read your request</p>
          <dl className="mt-1.5 grid gap-x-4 gap-y-1 sm:grid-cols-[auto_1fr]">
            {(
              [
                ["Genres", ir.preferredGenres],
                ["Tags", ir.preferredTags],
                ["Similar to", ir.similarTo],
                ["Avoid", ir.avoid],
              ] as const
            ).map(([label, values]) => (
              <div key={label} className="contents">
                <dt className="text-muted">{label}</dt>
                <dd className="text-paper">{values.length ? values.join(", ") : "None"}</dd>
              </div>
            ))}
          </dl>
        </div>
      ) : null}
    </div>
  );
}
