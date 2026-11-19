import { useEffect, useId, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ApiError, api, recommend } from "../api/client";
import type { EvaluationResults, RecommendationMode, RecommendationResponse } from "../api/types";
import { Alert } from "../components/Alert";
import { RankShift } from "../components/MovieCard";
import { EvaluationPanel } from "../components/EvaluationPanel";
import { UserPicker, UserSummaryCard, useUserSummary } from "../components/UserPicker";
import { Button, Panel, SectionHeading } from "../components/ui";
import { MODES, fmtMs } from "../lib/modes";
import { datasetReady, useStatus } from "../lib/status";

type Outcome =
  | { kind: "unavailable"; reason: string }
  | { kind: "pending" }
  | { kind: "running" }
  | { kind: "done"; response: RecommendationResponse }
  | { kind: "error"; error: ApiError };



export function ComparePage() {
  const { status } = useStatus();
  const [params, setParams] = useSearchParams();
  const initialUser = Number(params.get("user"));
  const [userId, setUserId] = useState<number | null>(Number.isInteger(initialUser) && initialUser > 0 ? initialUser : null);
  const [topN, setTopN] = useState(10);
  const [userRequest, setUserRequest] = useState("");
  const [outcomes, setOutcomes] = useState<Partial<Record<RecommendationMode, Outcome>>>({});
  const [running, setRunning] = useState(false);
  const [evaluation, setEvaluation] = useState<EvaluationResults | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);
  const { summary, error: summaryError } = useUserSummary(userId);
  const requestId = useId();
  const ready = datasetReady(status);

  useEffect(() => {
    api
      .evaluation()
      .then(setEvaluation)
      .catch((e: ApiError) => setEvalError(e.message));
  }, []);

  const run = async () => {
    if (!userId || !summary || !status) return;
    setRunning(true);
    const initial: Partial<Record<RecommendationMode, Outcome>> = {};
    for (const m of MODES) {
      initial[m.id] = status.modes[m.id]
        ? { kind: "pending" }
        : {
            kind: "unavailable",
            reason: m.id === "als_llm" && !status.llm.configured ? "LLM reranking is not configured." : "Model not trained.",
          };
    }
    setOutcomes(initial);
    // Sequential, so each model's response time is measured without contention.
    for (const m of MODES) {
      if (initial[m.id]?.kind !== "pending") continue;
      setOutcomes((prev) => ({ ...prev, [m.id]: { kind: "running" } }));
      try {
        const response = await recommend({ mode: m.id, userId, topN, userRequest });
        setOutcomes((prev) => ({ ...prev, [m.id]: { kind: "done", response } }));
      } catch (e) {
        setOutcomes((prev) => ({ ...prev, [m.id]: { kind: "error", error: e as ApiError } }));
      }
    }
    setRunning(false);
  };

  const done = MODES.filter((m) => outcomes[m.id]?.kind === "done").map((m) => ({
    mode: m,
    response: (outcomes[m.id] as { kind: "done"; response: RecommendationResponse }).response,
  }));
  const idSets = new Map(done.map((d) => [d.mode.id, new Set(d.response.recommendations.map((r) => r.movie.movieId))]));
  const countIn = (movieId: number) => [...idSets.values()].filter((s) => s.has(movieId)).length;
  const llm = done.find((d) => d.mode.id === "als_llm")?.response;

  return (
    <div className="space-y-12">
      <div>
        <h1 className="display text-4xl text-paper">Compare models</h1>
        <p className="measure mt-2 text-muted">
          Run every available model for the same MovieLens user and see where their lists agree, where they differ, and
          how the LLM reordered the Stage 1 shortlist.
        </p>
      </div>

      <Panel className="grid gap-6 p-5 md:grid-cols-[1fr_1fr] lg:grid-cols-[18rem_1fr_16rem]">
        <div className="space-y-4">
          <UserPicker
            selectedUserId={userId}
            onSelect={(id) => {
              setUserId(id);
              setOutcomes({});
              setParams({ user: String(id) }, { replace: true });
            }}
            disabled={!ready}
          />
          {summaryError ? <Alert tone="error" title="Invalid user" live>{summaryError.message}</Alert> : null}
        </div>
        <div className="space-y-4">
          {summary ? <UserSummaryCard summary={summary} compact /> : <p className="text-sm text-muted">Select a user to compare models.</p>}
        </div>
        <div className="space-y-4 md:col-span-2 lg:col-span-1">
          <fieldset>
            <legend className="mb-2 text-sm font-semibold text-paper">List length</legend>
            <div className="grid grid-cols-2 gap-1.5">
              {[5, 10].map((n) => (
                <label
                  key={n}
                  className={`flex min-h-11 cursor-pointer items-center justify-center rounded-[var(--radius-control)] border text-sm font-semibold has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-rose ${
                    topN === n ? "border-rose/70 bg-raised text-paper" : "border-line text-muted"
                  }`}
                >
                  <input type="radio" name="compare-topn" className="sr-only" checked={topN === n} onChange={() => setTopN(n)} />
                  Top {n}
                </label>
              ))}
            </div>
          </fieldset>
          <div>
            <label htmlFor={requestId} className="mb-1.5 block text-sm font-semibold text-paper">
              Request for the LLM mode (optional)
            </label>
            <textarea
              id={requestId}
              rows={2}
              maxLength={500}
              disabled={!status?.llm.configured}
              value={userRequest}
              onChange={(e) => setUserRequest(e.target.value)}
              placeholder={status?.llm.configured ? "Something light and funny" : "LLM reranking is not configured."}
              className="w-full rounded-[var(--radius-control)] border border-line bg-raised px-3 py-2 text-sm text-paper placeholder:text-faint disabled:opacity-60"
            />
          </div>
          <Button className="w-full" onClick={run} disabled={running || !summary || !ready}>
            {running ? "Comparing…" : "Compare models"}
          </Button>
        </div>
      </Panel>

      {Object.keys(outcomes).length ? (
        <section aria-labelledby="lists-title">
          <SectionHeading id="lists-title" sub="Movies marked “only here” appear in no other model's list.">
            Top {topN} from each model
          </SectionHeading>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {MODES.map((m) => {
              const outcome = outcomes[m.id];
              return (
                <Panel key={m.id} className="p-4">
                  <h3 className="flex items-center gap-2 font-semibold text-paper">
                    <span aria-hidden className="h-2.5 w-2.5 rounded-full" style={{ background: m.colorVar }} />
                    {m.label}
                  </h3>
                  {outcome?.kind === "done" ? (
                    <>
                      <p className="num mt-0.5 text-xs text-muted">
                        Response time {fmtMs(outcome.response.elapsedMs)}
                        {outcome.response.usedFallback ? ", ALS fallback" : ""}
                      </p>
                      {outcome.response.usedFallback ? (
                        <p className="mt-2 text-xs text-warn">
                          {outcome.response.fallbackLabel} {outcome.response.fallbackReason}
                        </p>
                      ) : null}
                      <ol className="mt-3 space-y-1.5 text-sm">
                        {outcome.response.recommendations.map((r) => (
                          <li key={r.movie.movieId} className="flex gap-2">
                            <span className="num w-5 shrink-0 text-right text-muted">{r.finalRank}</span>
                            <span className="min-w-0">
                              <span className="text-paper">{r.movie.title}</span>
                              {countIn(r.movie.movieId) === 1 && done.length > 1 ? (
                                <span className="ml-1.5 rounded-full border border-line px-1.5 text-[0.7rem] text-muted">only here</span>
                              ) : null}
                            </span>
                          </li>
                        ))}
                      </ol>
                    </>
                  ) : outcome?.kind === "error" ? (
                    <p className="mt-2 text-sm text-bad" role="alert">{outcome.error.message}</p>
                  ) : outcome?.kind === "unavailable" ? (
                    <p className="mt-2 text-sm text-muted">{outcome.reason}</p>
                  ) : (
                    <p className="mt-2 text-sm text-muted" role="status">
                      {outcome?.kind === "running" ? "Running…" : "Waiting…"}
                    </p>
                  )}
                </Panel>
              );
            })}
          </div>
        </section>
      ) : null}

      {done.length > 1 ? (
        <section aria-labelledby="overlap-title">
          <SectionHeading id="overlap-title" sub="Number of movies two models have in common, out of each list.">
            Overlap between models
          </SectionHeading>
          <div className="overflow-x-auto">
            <table className="num min-w-[32rem] border-collapse text-sm">
              <caption className="sr-only">Pairwise overlap of recommended movies</caption>
              <thead>
                <tr>
                  <th scope="col" className="p-2 text-left font-normal text-muted">Model</th>
                  {done.map((d) => (
                    <th key={d.mode.id} scope="col" className="p-2 text-right font-semibold text-paper">{d.mode.short}</th>
                  ))}
                  <th scope="col" className="p-2 text-right font-normal text-muted">Unique</th>
                </tr>
              </thead>
              <tbody>
                {done.map((row) => {
                  const a = idSets.get(row.mode.id)!;
                  const unique = [...a].filter((id) => countIn(id) === 1).length;
                  return (
                    <tr key={row.mode.id} className="border-t border-line">
                      <th scope="row" className="p-2 text-left font-semibold text-paper">{row.mode.short}</th>
                      {done.map((col) => {
                        const b = idSets.get(col.mode.id)!;
                        const shared = [...a].filter((id) => b.has(id)).length;
                        return (
                          <td key={col.mode.id} className={`p-2 text-right ${row.mode.id === col.mode.id ? "text-faint" : "text-paper"}`}>
                            {shared}
                          </td>
                        );
                      })}
                      <td className="p-2 text-right text-paper">{unique}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {llm && !llm.usedFallback ? (
        <section aria-labelledby="shift-title">
          <SectionHeading id="shift-title" sub="Where each LLM pick sat in the ALS candidate list before reranking.">
            Original ALS rank versus final LLM rank
          </SectionHeading>
          <div className="overflow-x-auto">
            <table className="num w-full min-w-[28rem] border-collapse text-sm">
              <thead>
                <tr className="text-muted">
                  <th scope="col" className="p-2 text-left font-normal">Movie</th>
                  <th scope="col" className="p-2 text-right font-normal">ALS rank</th>
                  <th scope="col" className="p-2 text-right font-normal">LLM rank</th>
                  <th scope="col" className="p-2 text-right font-normal">Change</th>
                </tr>
              </thead>
              <tbody>
                {llm.recommendations.map((r) => (
                  <tr key={r.movie.movieId} className="border-t border-line">
                    <td className="p-2 text-paper">{r.movie.title}</td>
                    <td className="p-2 text-right">#{r.originalAlsRank}</td>
                    <td className="p-2 text-right">#{r.finalRank}</td>
                    <td className="p-2 text-right"><RankShift from={r.originalAlsRank!} to={r.finalRank} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section aria-labelledby="eval-title">
        <SectionHeading id="eval-title" sub="Offline metrics from the evaluation command, computed on held-out ratings.">
          Evaluation
        </SectionHeading>
        {evalError ? (
          <Alert tone="error" title="Could not load evaluation results">{evalError}</Alert>
        ) : !evaluation ? (
          <p className="text-muted" role="status">Loading…</p>
        ) : (
          <EvaluationPanel evaluation={evaluation} />
        )}
      </section>
    </div>
  );
}
