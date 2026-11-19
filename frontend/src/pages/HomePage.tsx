import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, api } from "../api/client";
import type { DatasetStatistics } from "../api/types";
import { Alert } from "../components/Alert";
import { PipelineFunnel } from "../components/PipelineFunnel";
import { Poster } from "../components/Poster";
import { SectionHeading } from "../components/ui";
import { fmtDate, fmtInt } from "../lib/modes";
import { datasetReady, useStatus } from "../lib/status";

export function useStatistics() {
  const { status } = useStatus();
  const [stats, setStats] = useState<DatasetStatistics | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const ready = datasetReady(status);
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    api
      .statistics()
      .then((s) => !cancelled && setStats(s))
      .catch((e: ApiError) => !cancelled && setError(e));
    return () => {
      cancelled = true;
    };
  }, [ready]);
  return { stats, error, ready };
}

const WORKFLOW = [
  ["Select a MovieLens user", "Search the real user IDs and see what each person has rated."],
  ["Choose a recommendation mode", "Popularity, Item-KNN, ALS, or ALS with LLM reranking."],
  ["Optionally describe what you want", "Only used by the LLM mode, for example a mood or a movie to resemble."],
  ["Generate and rerank candidates", "ALS proposes candidates; the LLM can reorder them but never add new ones."],
  ["Receive Top-10 recommendations", "Each pick shows its scores, its original rank, and why it was chosen."],
];

export function HomePage() {
  const { stats, error, ready } = useStatistics();

  return (
    <div className="space-y-16">
      <section aria-labelledby="hero-title" className="grid items-center gap-10 pt-4 lg:grid-cols-[1fr_1.05fr]">
        <div className="min-w-0">
          <h1 id="hero-title" className="display text-[clamp(2.4rem,11vw,3.75rem)] text-paper">
            MovieMatch
          </h1>
          <p className="display-soft mt-4 text-xl text-paper/90 sm:text-2xl">
            Classical movie recommendations enhanced by LLM reranking
          </p>
          <p className="measure mt-4 text-muted">
            MovieMatch uses ALS to generate personalized candidates, Item-KNN to provide similarity evidence, and an
            LLM to understand your current request and rerank the results.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link
              to="/recommendations"
              className="inline-flex min-h-11 items-center rounded-[var(--radius-control)] border border-wine bg-wine px-5 text-sm font-semibold text-paper hover:bg-[#8e2648]"
            >
              Get Recommendations
            </Link>
            <a
              href="#how-it-works"
              className="inline-flex min-h-11 items-center rounded-[var(--radius-control)] border border-line bg-raised px-5 text-sm font-semibold text-paper hover:border-muted"
            >
              How It Works
            </a>
          </div>
        </div>
        <PipelineFunnel stats={stats} />
      </section>

      <section aria-labelledby="movies-title">
        <SectionHeading id="movies-title" sub="The most-rated movies in MovieLens 32M, read from the processed dataset.">
          Most rated in the dataset
        </SectionHeading>
        {stats ? (
          <ul className="-mx-4 flex snap-x gap-4 overflow-x-auto px-4 pb-3 sm:mx-0 sm:px-0">
            {stats.mostRatedMovies.map((m) => (
              <li key={m.movieId} className="w-36 shrink-0 snap-start sm:w-40">
                <Poster movie={m} />
                <p className="mt-2 line-clamp-2 text-sm font-semibold text-paper">{m.title}</p>
                <p className="num text-xs text-muted">
                  {m.year ?? "Year unknown"}, {fmtInt(m.ratingCount)} ratings
                </p>
              </li>
            ))}
          </ul>
        ) : error ? (
          <Alert tone="error" title="Could not load movies">{error.message}</Alert>
        ) : !ready ? (
          <p className="text-muted">Movies appear here once MovieLens 32M has been preprocessed.</p>
        ) : (
          <p className="text-muted" role="status">Loading movies…</p>
        )}
      </section>

      <section aria-labelledby="stats-title">
        <SectionHeading id="stats-title" sub="Counted from the files on this server after cleaning and deduplication.">
          MovieLens 32M at a glance
        </SectionHeading>
        {stats ? (
          <>
            <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-[var(--radius-card)] border border-line bg-line sm:grid-cols-3 lg:grid-cols-6">
              {[
                ["Ratings", fmtInt(stats.ratings)],
                ["Users", fmtInt(stats.users)],
                ["Movies", fmtInt(stats.movies)],
                ["Tag applications", fmtInt(stats.tagApplications)],
                ["Mean rating", stats.meanRating.toFixed(2)],
                ["Ratings span", `${fmtDate(stats.firstRatingAt)} to ${fmtDate(stats.lastRatingAt)}`],
              ].map(([label, value]) => (
                <div key={label} className="bg-panel p-4">
                  <dt className="text-sm text-muted">{label}</dt>
                  <dd className="num mt-1 text-lg font-semibold text-paper">{value}</dd>
                </div>
              ))}
            </dl>
            <p className="mt-3 text-sm text-muted">
              Chronological split per user: {fmtInt(stats.split.train)} training, {fmtInt(stats.split.validation)}{" "}
              validation, {fmtInt(stats.split.test)} test interactions.
              {stats.matchesPublishedMl32mCounts === false
                ? " Row counts differ from the published ml-32m figures; check MOVIELENS_DATA_PATH."
                : ""}
            </p>
          </>
        ) : (
          <p className="text-muted">
            {ready ? "Loading statistics…" : "Statistics appear once the dataset has been preprocessed."}
          </p>
        )}
      </section>

      <section aria-labelledby="how-it-works" className="scroll-mt-6">
        <SectionHeading id="how-it-works">How it works</SectionHeading>
        <div className="grid gap-10 lg:grid-cols-2">
          <ol className="space-y-4">
            {WORKFLOW.map(([title, text], i) => (
              <li key={title} className="flex gap-4">
                <span
                  aria-hidden
                  className="display num flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-line text-sm text-rose"
                >
                  {i + 1}
                </span>
                <div>
                  <p className="font-semibold text-paper">{title}</p>
                  <p className="text-sm text-muted">{text}</p>
                </div>
              </li>
            ))}
          </ol>
          <div className="measure space-y-3 text-muted">
            <p>
              <span className="text-paper">ALS</span> factorizes the training ratings into user and movie vectors and
              scores unseen movies for the user's rating history.
            </p>
            <p>
              <span className="text-paper">Item-KNN</span> scores similar movies. Its picks are combined with ALS picks
              into a 50-movie shortlist.
            </p>
            <p>
              <span className="text-paper">The LLM</span> receives only those candidates with their MovieLens genres,
              tags, and scores. It reorders them for your request and explains each pick. The backend rejects any
              response that names a movie outside the candidate list and falls back to the Stage 1 order if needed.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
