import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { PipelineFunnel } from "../components/PipelineFunnel";
import { Panel } from "../components/ui";
import { useStatus } from "../lib/status";
import { useStatistics } from "./HomePage";

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="measure space-y-3">
      <h2 id={id} className="display-soft text-2xl text-paper">{title}</h2>
      <div className="space-y-3 text-muted [&_strong]:font-semibold [&_strong]:text-paper">{children}</div>
    </section>
  );
}

export function AboutPage() {
  const { status } = useStatus();
  const { stats } = useStatistics();
  const llm = status?.llm;

  return (
    <div className="space-y-12">
      <div className="measure">
        <h1 className="display text-4xl text-paper">About MovieMatch</h1>
        <p className="mt-3 text-muted">
          A course project that asks a narrow question: if a classical recommender has already found good candidates,
          can a language model improve the order and explain it, without being allowed to invent anything?
        </p>
      </div>

      <Section id="purpose" title="Project purpose">
        <p>
          Classical collaborative filtering is strong at finding movies a person is likely to enjoy, but it cannot read a
          request like “something like Interstellar, with less action” and its scores are hard to explain. MovieMatch keeps
          the classical models in charge of <strong>which</strong> movies are considered and lets an LLM decide only the{" "}
          <strong>order</strong> and the <strong>explanation</strong>, using data the backend supplies.
        </p>
      </Section>

      <section aria-labelledby="architecture" className="grid gap-8 lg:grid-cols-[1fr_1fr]">
        <div className="measure space-y-3">
          <h2 id="architecture" className="display-soft text-2xl text-paper">System architecture</h2>
          <ol className="list-decimal space-y-1.5 pl-5 text-muted marker:text-faint">
            <li>
              MovieLens 32M is split by date, as in the proposal: ratings before 2020 train the models, 2020–21 is for
              tuning (validation), and 2022–23 is for the final test.
            </li>
            <li>
              Stage 1: ALS and Item-KNN each rank every unseen movie; their top 30s are merged into a 50-movie
              shortlist. (The LLM mode currently reranks ALS's top 50.)
            </li>
            <li>Item-KNN attaches similarity evidence linking each candidate to movies the user rated highly.</li>
            <li>Candidate genres, tags, scores, and the user's top-rated movies are collected.</li>
            <li>The LLM interprets the request and reranks only the supplied candidates.</li>
            <li>The backend validates the response and returns the final list with explanations.</li>
          </ol>
          <p className="text-sm text-muted">
            Frontend: React, TypeScript, Tailwind CSS. Backend: FastAPI, pandas, SciPy sparse matrices, the implicit
            library for ALS and item similarity, Pydantic for validation.
          </p>
        </div>
        <PipelineFunnel stats={stats} />
      </section>

      <Section id="models" title="The models">
        <p>
          All models learn only from ratings before the evaluated period and never recommend a movie the user has
          already rated.
        </p>
        <p>
          <strong>Popularity.</strong> The proposal's baseline: the most-rated movies, the same list for everyone. A
          second version, <strong>Popularity (weighted)</strong>, ranks by a Bayesian weighted rating, WR = v/(v+m)·R +
          m/(v+m)·C, so movies that are both widely and well rated come first.
        </p>
        <p>
          <strong>Item-KNN.</strong> Cosine similarity between movies' rating vectors, keeping each movie's 50 nearest
          neighbours in a sparse matrix (never a dense 87,585 × 87,585 one). The app scores a movie by the sum of its
          similarities to the movies the user rated 4.0 or higher, weighted by those ratings. The proposal's formula, a
          similarity-weighted average of the user's ratings, is also evaluated. On the validation period it scored far
          lower, because averages favour obscure movies with one or two strong neighbours, so the sum is the default.
        </p>
        <p>
          <strong>ALS.</strong> Matrix factorization. The default version treats ratings as implicit feedback: each rated
          movie is a positive interaction with confidence 1 + α·rating. <strong>ALS (explicit ratings)</strong> is the
          proposal's version: it learns to predict the star rating itself (squared error on observed ratings, with the
          same regularisation as Spark MLlib) and ranks movies by predicted rating. When serving, a user's vector is
          re-solved from their ratings with the movie vectors held fixed. Displayed ALS scores are min-max normalized
          across the user's candidates: relative scores, not probabilities.
        </p>
        <p>
          <strong>Stage 1 (ALS + KNN merged).</strong> The proposal's shortlist: ALS's top 30 and Item-KNN's top 30 are
          combined, scores rescaled to 0–1 per model and averaged where both picked a movie, and topped up from ALS to 50.
          The shortlist is what the LLM stage reorders.
        </p>
        <p>
          <strong>NeuMF.</strong> Neural collaborative filtering (He et al., 2017), the deep-learning baseline: a
          matrix-factorization branch and a multi-layer network branch trained on "rated it" signals with sampled
          negatives. It makes its own top-10 list and is compared with the same metrics; it does not feed the LLM stage.
        </p>
      </Section>

      <Section id="reranking" title="LLM reranking">
        <p>
          <strong>
            Reranking changes the order of the combined ALS and Item-KNN shortlist. The LLM cannot add movies
            outside that shortlist.
          </strong>
        </p>
        <p>
          The backend sends the request text, the user's highly rated movies and favourite genres, and for each candidate
          its title, year, MovieLens genres and tags, Stage 1 and model scores, and up to two KNN similarities. The LLM must
          return strict JSON. The backend checks that every movie ID came from the candidate list, that there are no
          duplicates, that ranks run 1 to N, that scores are numbers between 0 and 1, and that explanations are short. One
          correction attempt is allowed; if it also fails, or the LLM times out, the page shows the Stage 1 order marked
          “Stage 1 fallback because LLM reranking failed.”
        </p>
      </Section>

      <Section id="dataset" title="MovieLens 32M">
        <p>
          32 million ratings and 2 million tag applications for 87,585 movies by 200,948 users, collected through
          October 2023 and released by GroupLens in May 2024 as a stable benchmark. Ratings use a 0.5 to 5 star scale.
          The dataset is read from the server; it is never loaded in the browser.
        </p>
        <ul className="space-y-1">
          <li>
            <a className="text-rose underline underline-offset-2" href="https://grouplens.org/datasets/movielens/32m/" target="_blank" rel="noreferrer">
              Dataset page
            </a>
          </li>
          <li>
            <a className="text-rose underline underline-offset-2" href="https://files.grouplens.org/datasets/movielens/ml-32m-README.html" target="_blank" rel="noreferrer">
              README and license
            </a>
          </li>
          <li>
            <a className="text-rose underline underline-offset-2" href="https://files.grouplens.org/datasets/movielens/ml-32m.zip" target="_blank" rel="noreferrer">
              Download ml-32m.zip (about 239 MB)
            </a>
          </li>
        </ul>
      </Section>

      <section aria-labelledby="llm-status" className="measure space-y-3">
        <h2 id="llm-status" className="display-soft text-2xl text-paper">Current LLM configuration</h2>
        <Panel className="p-4">
          {!status ? (
            <p className="text-muted">Status unavailable: the backend could not be reached.</p>
          ) : llm?.configured ? (
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
              <dt className="text-muted">Status</dt>
              <dd className="text-ok">Configured</dd>
              <dt className="text-muted">Model</dt>
              <dd className="text-paper [overflow-wrap:anywhere]">{llm.model}</dd>
              <dt className="text-muted">API key</dt>
              <dd className="text-paper">{llm.apiKeySet ? "Set on the server" : "Not set (not required by every provider)"}</dd>
              <dt className="text-muted">Timeout</dt>
              <dd className="num text-paper">{llm.timeoutSeconds} s</dd>
              <dt className="text-muted">JSON mode</dt>
              <dd className="text-paper">{llm.jsonMode ? "Requested" : "Off"}</dd>
            </dl>
          ) : (
            <div className="space-y-1 text-sm">
              <p className="text-paper">LLM reranking is not configured.</p>
              <p className="text-muted">
                Missing: {llm?.missing.join(", ") || "unknown"}. Popularity, Item-KNN, and ALS are unaffected. Any
                OpenAI-compatible endpoint works, set through environment variables on the server.
              </p>
            </div>
          )}
        </Panel>
      </section>

      <Section id="limitations" title="Limitations">
        <ul className="list-disc space-y-1.5 pl-5 marker:text-faint">
          <li>MovieLens has no posters, plots, cast, or directors. Cards use generated placeholders unless a poster source is configured, and explanations can only cite genres, tags, and ratings.</li>
          <li>The LLM can only reorder 50 candidates. If ALS misses a movie that fits the request, reranking cannot recover it.</li>
          <li>Stage 1 + LLM reranking is not part of the offline evaluation, because it would need one LLM call per evaluated user.</li>
          <li>LLM explanations are constrained and validated for structure, but their wording is not fact-checked sentence by sentence.</li>
          <li>Tags are community-applied and uneven: many movies have none.</li>
          <li>MovieLens users are not a random sample of viewers, and ratings stop in October 2023.</li>
          <li>With the date split, few users are cold-start (1–10 earlier ratings), so cold-start results rest on a small group; the Compare page shows how many.</li>
          <li>Movies first rated after 2019 have no earlier ratings, so no rating-based model can recommend them in the test period.</li>
        </ul>
        <p>
          <Link to="/recommendations" className="text-rose underline underline-offset-2">Try the recommender</Link>
        </p>
      </Section>
    </div>
  );
}
