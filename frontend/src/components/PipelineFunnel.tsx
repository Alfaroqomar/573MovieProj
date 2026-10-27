import type { DatasetStatistics } from "../api/types";
import { fmtInt } from "../lib/modes";

interface Stage {
  title: string;
  detail: string;
  count: number | null;
  unit: string;
}

/**
 * The recommendation pipeline drawn as a funnel. Bar widths are log-scaled
 * from real counts in the processed dataset; the LLM stage keeps the same
 * width as the candidate stage because it can only reorder those movies.
 */
export function PipelineFunnel({
  stats,
  candidates = 50,
  finalCount = 10,
}: {
  stats: DatasetStatistics | null;
  candidates?: number;
  finalCount?: number;
}) {
  const stages: Stage[] = [
    {
      title: "User ratings",
      detail: stats ? `from ${fmtInt(stats.users)} MovieLens users` : "from MovieLens 32M",
      count: stats?.ratings ?? null,
      unit: "ratings",
    },
    {
      title: "Classical models",
      detail: "Popularity, Item-KNN, and ALS score every unseen movie",
      count: stats?.ratedMovies ?? null,
      unit: "rated movies",
    },
    {
      title: "Candidate movies",
      detail: "ALS keeps its top unseen movies (20 to 100; 50 by default)",
      count: candidates,
      unit: "candidates",
    },
    {
      title: "LLM assistance",
      detail: "Interprets your request, reranks only these candidates, writes grounded explanations",
      count: candidates,
      unit: "same candidates",
    },
    {
      title: "Validated recommendations",
      detail: "IDs, ranks, and scores checked by the backend before display",
      count: finalCount,
      unit: "final picks",
    },
  ];
  const max = Math.log10(Math.max(stats?.ratings ?? 32_000_000, 10));
  const width = (n: number | null) => (n == null ? 100 : 26 + 74 * (Math.log10(Math.max(n, 1)) / max));

  return (
    <ol aria-label="MovieMatch recommendation pipeline" className="space-y-2.5">
      {stages.map((stage, i) => (
        <li key={stage.title}>
          <div
            className="rounded-[10px] border px-3.5 py-2.5"
            style={{
              width: `${width(stage.count)}%`,
              minWidth: "min(100%, 15rem)",
              background: `color-mix(in srgb, var(--color-wine) ${78 - i * 12}%, var(--color-panel))`,
              borderColor: `color-mix(in srgb, var(--color-rose) ${30 - i * 4}%, var(--color-line))`,
            }}
          >
            <div className="flex flex-wrap items-baseline justify-between gap-x-3">
              <span className="font-semibold text-paper">{stage.title}</span>
              <span className="num text-sm text-paper/85">
                {stage.count != null ? `${fmtInt(stage.count)} ${stage.unit}` : ""}
              </span>
            </div>
            <p className="text-sm text-paper/75">{stage.detail}</p>
          </div>
        </li>
      ))}
    </ol>
  );
}
