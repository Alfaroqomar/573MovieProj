import type { Recommendation } from "../api/types";
import { fmtInt, fmtScore, modeInfo } from "../lib/modes";
import { Poster } from "./Poster";
import { Button, Facts, GenreList, ModeBadge } from "./ui";

export function RankShift({ from, to }: { from: number; to: number }) {
  const moved = from - to;
  const text =
    moved > 0 ? `up ${moved} from Stage 1 #${from}` : moved < 0 ? `down ${-moved} from Stage 1 #${from}` : `same as Stage 1 #${from}`;
  return (
    <span className="num inline-flex items-center gap-1 text-xs text-muted">
      <span aria-hidden className={moved > 0 ? "text-ok" : moved < 0 ? "text-bad" : "text-faint"}>
        {moved > 0 ? "▲" : moved < 0 ? "▼" : "●"}
      </span>
      {text}
    </span>
  );
}

const SOURCE_LABELS: Record<string, string> = {
  both: "ALS and Item-KNN",
  als: "ALS",
  knn: "Item-KNN",
  als_top_up: "ALS (top-up)",
};

export function scoreRows(rec: Recommendation): Array<[string, string] | false> {
  if (rec.usedFallback && rec.mode === "neumf") {
    return [["Training ratings", rec.ratingCount != null ? fmtInt(rec.ratingCount) : "n/a"]];
  }
  switch (rec.mode) {
    case "popularity":
      return [
        ["Training ratings", rec.ratingCount != null ? fmtInt(rec.ratingCount) : "n/a"],
        ["Mean rating", fmtScore(rec.meanRating)],
      ];
    case "popularity_weighted":
      return [
        ["Weighted rating", fmtScore(rec.popularityScore)],
        ["Training ratings", rec.ratingCount != null ? fmtInt(rec.ratingCount) : "n/a"],
        ["Mean rating", fmtScore(rec.meanRating)],
      ];
    case "item_knn":
      return [
        ["KNN score", fmtScore(rec.knnScore)],
        ["Strongest similarity", fmtScore(rec.knnEvidence?.[0]?.similarity)],
      ];
    case "als":
      return [
        ["ALS rank", `#${rec.originalAlsRank}`],
        ["ALS score", fmtScore(rec.alsScore)],
      ];
    case "als_explicit":
      return [
        ["Predicted rating", rec.predictedRating != null ? `${Math.min(rec.predictedRating, 5).toFixed(2)} / 5` : "n/a"],
        ["Relative score", fmtScore(rec.alsScore)],
      ];
    case "stage1":
      return [
        ["Shortlist rank", `#${rec.finalRank} of 50`],
        ["Picked by", SOURCE_LABELS[rec.candidateSource ?? ""] ?? "n/a"],
        ["Merged score", fmtScore(rec.mergedScore)],
        rec.originalAlsRank != null && ["ALS rank", `#${rec.originalAlsRank}`],
        rec.knnRank != null && ["Item-KNN rank", `#${rec.knnRank}`],
      ];
    case "neumf":
      return [["NeuMF score", fmtScore(rec.neumfScore)]];
    case "als_llm":
      return [
        ["Stage 1 rank", `#${rec.stage1Rank}`],
        ["Final rank", `#${rec.finalRank}`],
        ["Merged score", fmtScore(rec.mergedScore)],
        ["Strongest KNN similarity", fmtScore(rec.strongestKnnSimilarity)],
        !rec.usedFallback && ["LLM request match", fmtScore(rec.llmRelevanceScore)],
      ];
  }
}

export function MovieCard({ rec, onDetails }: { rec: Recommendation; onDetails: (rec: Recommendation) => void }) {
  const info = modeInfo(rec.mode);
  const titleId = `rec-${rec.mode}-${rec.movie.movieId}`;
  const showShift = rec.mode === "als_llm" && !rec.usedFallback && rec.stage1Rank != null;
  const evidence = rec.knnEvidence ?? [];

  return (
    <article
      aria-labelledby={titleId}
      className="flex w-full flex-col gap-4 rounded-[var(--radius-card)] border border-line bg-panel p-4 shadow-[var(--shadow-card)]"
    >
      <div className="flex items-start justify-between gap-3">
        <p className="display num text-4xl leading-none text-paper" aria-label={`Rank ${rec.finalRank}`}>
          {rec.finalRank}
        </p>
        <div className="flex flex-col items-end gap-1">
          <ModeBadge label={info.short} color={info.colorVar} />
          {showShift ? <RankShift from={rec.stage1Rank!} to={rec.finalRank} /> : null}
        </div>
      </div>

      <div className="flex gap-3">
        <div className="w-20 shrink-0">
          <Poster movie={rec.movie} compact />
        </div>
        <div className="min-w-0 space-y-2">
          <h3 id={titleId} className="display-soft text-lg text-paper [overflow-wrap:anywhere]">
            {rec.movie.title}
          </h3>
          <p className="num text-sm text-muted">{rec.movie.year ?? "Release year unknown"}</p>
          <GenreList genres={rec.movie.genres} small />
        </div>
      </div>

      <Facts rows={scoreRows(rec)} />

      {rec.mode === "item_knn" && evidence.length > 0 ? (
        <div className="text-sm">
          <p className="text-muted">Similar to movies you rated highly</p>
          <ul className="mt-1 space-y-0.5">
            {evidence.slice(0, 3).map((e) => (
              <li key={e.historyMovieId} className="flex justify-between gap-2">
                <span className="truncate text-paper">{e.historyMovieTitle}</span>
                <span className="num shrink-0 text-muted">{e.similarity.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {rec.explanation ? (
        <div className="text-sm">
          <p className="text-muted">{rec.explanationSource === "llm" ? "Why (LLM)" : "Why"}</p>
          <p className="mt-0.5 text-paper">{rec.explanation}</p>
        </div>
      ) : null}

      <div className="mt-auto">
        <Button variant="secondary" className="w-full" onClick={() => onDetails(rec)}>
          View details<span className="sr-only"> for {rec.movie.title}</span>
        </Button>
      </div>
    </article>
  );
}
