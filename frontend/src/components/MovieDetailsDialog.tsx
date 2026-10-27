import { useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api/client";
import type { MovieDetails, Recommendation } from "../api/types";
import { fmtInt, fmtScore, modeInfo } from "../lib/modes";
import { Alert } from "./Alert";
import { scoreRows } from "./MovieCard";
import { Poster } from "./Poster";
import { Button, Facts, GenreList, ModeBadge } from "./ui";

/**
 * Accessible modal built on the native <dialog> element (focus is moved into
 * the dialog, trapped while open, and restored on close; Escape closes it).
 */
export function MovieDetailsDialog({ rec, onClose }: { rec: Recommendation | null; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const [details, setDetails] = useState<MovieDetails | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (rec && !dialog.open) dialog.showModal();
    if (!rec && dialog.open) dialog.close();
  }, [rec]);

  useEffect(() => {
    if (!rec) return;
    let cancelled = false;
    setDetails(null);
    setError(null);
    api
      .movie(rec.movie.movieId)
      .then((d) => !cancelled && setDetails(d))
      .catch((e: ApiError) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [rec]);

  const movie = details ?? rec?.movie;
  const info = rec ? modeInfo(rec.mode) : null;

  return (
    <dialog
      ref={ref}
      aria-labelledby="details-title"
      onClose={onClose}
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
      className="m-auto max-h-[92dvh] w-[min(46rem,calc(100vw-1.5rem))] overflow-y-auto rounded-[var(--radius-card)] border border-line bg-panel p-0 text-paper shadow-2xl backdrop:bg-black/70"
    >
      {rec && movie && info ? (
        <div className="p-5 sm:p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <ModeBadge label={info.label} color={info.colorVar} />
              <h2 id="details-title" className="display mt-2 text-2xl text-paper sm:text-3xl [overflow-wrap:anywhere]">
                {movie.title}
              </h2>
              <p className="num mt-1 text-muted">{movie.year ?? "Release year unknown"}</p>
            </div>
            <Button variant="ghost" onClick={onClose} aria-label="Close details">
              Close
            </Button>
          </div>

          <div className="mt-5 grid gap-6 sm:grid-cols-[10rem_1fr]">
            <div className="mx-auto w-36 sm:w-full">
              <Poster movie={movie} />
            </div>
            <div className="space-y-5">
              <section aria-labelledby="d-genres">
                <h3 id="d-genres" className="mb-1.5 text-sm font-semibold text-paper">Genres</h3>
                <GenreList genres={movie.genres} />
              </section>
              <section aria-labelledby="d-tags">
                <h3 id="d-tags" className="mb-1.5 text-sm font-semibold text-paper">MovieLens tags</h3>
                {movie.tags.length ? (
                  <p className="text-sm text-muted">{movie.tags.join(", ")}</p>
                ) : (
                  <p className="text-sm text-muted">No tags have been applied to this movie.</p>
                )}
              </section>
              <section aria-labelledby="d-scores">
                <h3 id="d-scores" className="mb-1.5 text-sm font-semibold text-paper">Scores and ranks</h3>
                <Facts
                  rows={[
                    ["Final rank", `#${rec.finalRank}`],
                    ...scoreRows(rec).filter((r) => r && r[0] !== "Final rank"),
                    rec.alsRawScore != null && ["Raw ALS score", rec.alsRawScore.toFixed(4)],
                    details != null && ["MovieLens ratings (all splits)", fmtInt(details.ratingCount)],
                    details?.meanRating != null && ["Mean MovieLens rating", fmtScore(details.meanRating)],
                  ]}
                />
              </section>
            </div>
          </div>

          <section aria-labelledby="d-history" className="mt-6">
            <h3 id="d-history" className="mb-1.5 text-sm font-semibold text-paper">Similar movies from this user's history</h3>
            {rec.knnEvidence && rec.knnEvidence.length ? (
              <ul className="divide-y divide-line rounded-[var(--radius-control)] border border-line">
                {rec.knnEvidence.map((e) => (
                  <li key={e.historyMovieId} className="flex items-center justify-between gap-3 px-3 py-2 text-sm">
                    <span className="min-w-0 truncate text-paper">{e.historyMovieTitle}</span>
                    <span className="num shrink-0 text-muted">
                      rated {e.historyRating.toFixed(1)}, similarity {e.similarity.toFixed(3)}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted">
                {rec.mode === "popularity"
                  ? "The popularity baseline does not use the user's history beyond excluding movies already rated."
                  : "No item-similarity evidence links this movie to the user's highly rated movies."}
              </p>
            )}
          </section>

          <section aria-labelledby="d-why" className="mt-6">
            <h3 id="d-why" className="mb-1.5 text-sm font-semibold text-paper">
              {rec.explanationSource === "llm" ? "Explanation from the LLM" : "Explanation"}
            </h3>
            <p className="text-paper">{rec.explanation}</p>
            <p className="mt-2 text-sm text-muted">
              {rec.explanationSource === "llm"
                ? "Generated from the genres, tags, scores, and history supplied by the backend, then validated."
                : rec.usedFallback
                  ? "ALS fallback because LLM reranking failed. This explanation is built from the ALS rank and similarity data."
                  : "Built by the backend from the model's scores and the user's ratings."}
            </p>
          </section>

          {error ? (
            <div className="mt-5">
              <Alert tone="warning" title="Some movie details could not be loaded">{error}</Alert>
            </div>
          ) : null}

          {details && (details.imdbId || details.tmdbId) ? (
            <p className="mt-6 text-sm text-muted">
              External pages from MovieLens links.csv:{" "}
              {details.imdbId ? (
                <a className="text-rose underline underline-offset-2" href={`https://www.imdb.com/title/${details.imdbId}/`} target="_blank" rel="noreferrer">
                  IMDb
                </a>
              ) : null}
              {details.imdbId && details.tmdbId ? ", " : null}
              {details.tmdbId ? (
                <a className="text-rose underline underline-offset-2" href={`https://www.themoviedb.org/movie/${details.tmdbId}`} target="_blank" rel="noreferrer">
                  TMDB
                </a>
              ) : null}
            </p>
          ) : null}
        </div>
      ) : null}
    </dialog>
  );
}
