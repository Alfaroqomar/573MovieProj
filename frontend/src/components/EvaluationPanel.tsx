import { useState } from "react";
import type { EvaluationGroup, EvaluationResults, MetricRow } from "../api/types";
import { fmtInt } from "../lib/modes";
import { Alert } from "./Alert";

const GROUPS: Array<{ id: EvaluationGroup; label: string }> = [
  { id: "all", label: "All users" },
  { id: "cold_start", label: "Cold-start users" },
  { id: "regular", label: "Regular users" },
];

const MODEL_ORDER = [
  "popularity",
  "popularity_weighted",
  "item_knn",
  "item_knn_sum",
  "als",
  "als_explicit",
  "stage1",
  "neumf",
  "stage1_rerank",
];

type Column = { key: string; label: string; digits: number; lowerIsBetter?: boolean; hint: string };

const ACCURACY: Column[] = [
  { key: "ndcg@10", label: "NDCG@10", digits: 4, hint: "Liked movies near the top of the list" },
  { key: "recall@10", label: "Recall@10", digits: 4, hint: "Share of liked movies found in the top 10" },
  { key: "precision@10", label: "Precision@10", digits: 4, hint: "Share of the 10 picks the user liked" },
  { key: "hitRate@10", label: "HitRate@10", digits: 3, hint: "Users with at least one liked pick" },
  { key: "mrr@10", label: "MRR@10", digits: 4, hint: "How early the first liked pick appears" },
  { key: "recall@50", label: "Recall@50", digits: 4, hint: "Liked movies in the top 50 (the LLM shortlist ceiling)" },
];

const BEYOND: Column[] = [
  { key: "coverage@10", label: "Coverage@10", digits: 4, hint: "Share of the catalog ever recommended" },
  { key: "gini@10", label: "Gini@10", digits: 3, lowerIsBetter: true, hint: "0 = spread evenly, 1 = same few movies for everyone" },
  { key: "serendipity@10", label: "Serendipity@10", digits: 4, hint: "Liked picks outside the 50 most-rated" },
  { key: "novelty@10", label: "Novelty@10", digits: 2, hint: "Higher = less popular movies (bits)" },
];

function order(names: string[]): string[] {
  return [...names].sort((a, b) => {
    const ia = MODEL_ORDER.indexOf(a);
    const ib = MODEL_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
}

function best(rows: Record<string, MetricRow>, col: Column): number | null {
  const values = Object.values(rows)
    .map((r) => r[col.key])
    .filter((v): v is number => typeof v === "number");
  if (!values.length) return null;
  return col.lowerIsBetter ? Math.min(...values) : Math.max(...values);
}

function MetricTable({
  caption,
  columns,
  rows,
  labels,
  ci,
  latency,
}: {
  caption: string;
  columns: Column[];
  rows: Record<string, MetricRow>;
  labels: Record<string, string>;
  ci?: Record<string, Record<string, [number, number]>>;
  latency?: Record<string, number>;
}) {
  const names = order(Object.keys(rows).filter((n) => Object.keys(rows[n] ?? {}).length));
  const bests = Object.fromEntries(columns.map((c) => [c.key, best(rows, c)]));
  return (
    <div className="overflow-x-auto rounded-[var(--radius-card)] border border-line">
      <table className="num w-full min-w-[40rem] border-collapse text-sm">
        <caption className="px-3 pt-3 text-left text-sm font-semibold text-paper">{caption}</caption>
        <thead>
          <tr className="text-muted">
            <th scope="col" className="p-2 pl-3 text-left font-normal">Model</th>
            {columns.map((c) => (
              <th key={c.key} scope="col" className="p-2 text-right font-normal" title={c.hint}>
                {c.label}
              </th>
            ))}
            {latency ? <th scope="col" className="p-2 pr-3 text-right font-normal">ms / user</th> : null}
          </tr>
        </thead>
        <tbody>
          {names.map((name) => (
            <tr key={name} className="border-t border-line">
              <th scope="row" className="p-2 pl-3 text-left font-semibold text-paper">{labels[name] ?? name}</th>
              {columns.map((c) => {
                const v = rows[name][c.key];
                const isBest = v != null && bests[c.key] != null && Math.abs(v - (bests[c.key] as number)) < 1e-9;
                const interval = ci?.[name]?.[c.key];
                return (
                  <td key={c.key} className={`p-2 text-right ${isBest ? "font-semibold text-paper" : "text-paper/85"}`}>
                    {v != null ? v.toFixed(c.digits) : "n/a"}
                    {interval ? (
                      <span className="block text-[0.7rem] font-normal text-muted">
                        {interval[0].toFixed(3)}–{interval[1].toFixed(3)}
                      </span>
                    ) : null}
                  </td>
                );
              })}
              {latency ? <td className="p-2 pr-3 text-right text-muted">{latency[name]?.toFixed(1) ?? "n/a"}</td> : null}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function EvaluationPanel({ evaluation }: { evaluation: EvaluationResults }) {
  const [group, setGroup] = useState<EvaluationGroup>("all");
  if (!evaluation.available) {
    return (
      <Alert tone="info" title={evaluation.message ?? "Evaluation not run yet."}>
        Run the evaluation command in the README to compute the accuracy and beyond-accuracy metrics.
      </Alert>
    );
  }
  const groups = evaluation.groups ?? { all: evaluation.models ?? {}, cold_start: {}, regular: {} };
  const rows = groups[group] ?? {};
  const labels = evaluation.modelLabels ?? {};
  const sizes = evaluation.groupSizes;
  const buckets = Object.entries(evaluation.historyBuckets ?? {});
  const unreachable = evaluation.relevantUnreachable?.[group];
  const empty = !Object.values(rows).some((r) => Object.keys(r).length);

  return (
    <div className="space-y-5">
      <div role="radiogroup" aria-label="User group" className="flex flex-wrap gap-1.5">
        {GROUPS.map((g) => (
          <button
            key={g.id}
            type="button"
            role="radio"
            aria-checked={group === g.id}
            onClick={() => setGroup(g.id)}
            className={`min-h-10 rounded-[var(--radius-control)] border px-3 text-sm ${
              group === g.id ? "border-rose/70 bg-raised text-paper" : "border-line text-muted hover:text-paper"
            }`}
          >
            {g.label}
            {sizes ? <span className="num ml-1.5 text-xs text-muted">{fmtInt(sizes[g.id] ?? 0)}</span> : null}
          </button>
        ))}
      </div>

      {empty ? (
        <Alert tone="info" title="No users in this group">
          The evaluated period has no eligible users in this group.
        </Alert>
      ) : (
        <>
          <MetricTable
            caption="Accuracy (best value in each column in bold; 95% bootstrap interval under NDCG and Recall)"
            columns={ACCURACY}
            rows={rows}
            labels={labels}
            ci={evaluation.confidence?.[group]}
            latency={evaluation.meanLatencyMs}
          />
          <MetricTable caption="Beyond accuracy" columns={BEYOND} rows={rows} labels={labels} />
        </>
      )}

      {buckets.length ? (
        <div className="overflow-x-auto rounded-[var(--radius-card)] border border-line">
          <table className="num w-full min-w-[32rem] border-collapse text-sm">
            <caption className="px-3 pt-3 text-left text-sm font-semibold text-paper">
              NDCG@10 by how many ratings the user had before the evaluated period
            </caption>
            <thead>
              <tr className="text-muted">
                <th scope="col" className="p-2 pl-3 text-left font-normal">Model</th>
                {buckets.map(([label, b]) => (
                  <th key={label} scope="col" className="p-2 text-right font-normal">
                    {label} ratings
                    <span className="block text-[0.7rem]">{fmtInt(b.users)} users</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {order(Object.keys(buckets[0][1].models)).map((name) => (
                <tr key={name} className="border-t border-line">
                  <th scope="row" className="p-2 pl-3 text-left font-semibold text-paper">{labels[name] ?? name}</th>
                  {buckets.map(([label, b]) => (
                    <td key={label} className="p-2 text-right text-paper/85">
                      {b.models[name]?.["ndcg@10"]?.toFixed(4) ?? "n/a"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <div className="measure space-y-1.5 text-sm text-muted">
        <p>
          {evaluation.split === "test" ? "Test" : "Validation"} period. {fmtInt(evaluation.usersEvaluated ?? 0)} users
          sampled from {fmtInt(evaluation.eligibleUsers ?? 0)} eligible
          {evaluation.coldStartExtraUsers
            ? `, plus ${fmtInt(evaluation.coldStartExtraUsers)} extra cold-start users for that group`
            : ""}
          . Cold-start means {evaluation.coldStartDefinition ?? "few ratings"}. {evaluation.protocol}
        </p>
        {unreachable != null ? (
          <p>
            {(unreachable * 100).toFixed(1)}% of liked movies in this group had no earlier ratings, so no rating-based
            model could recommend them.
          </p>
        ) : null}
        {evaluation.skippedModels && Object.keys(evaluation.skippedModels).length ? (
          <p>
            Not included:{" "}
            {Object.entries(evaluation.skippedModels)
              .map(([n, why]) => `${labels[n] ?? n} (${why})`)
              .join(", ")}
            .
          </p>
        ) : null}
        {evaluation.notEvaluated?.als_llm ? <p>ALS + LLM: {evaluation.notEvaluated.als_llm}</p> : null}
      </div>
    </div>
  );
}
