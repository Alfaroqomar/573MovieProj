import { useState } from "react";
import type { Movie } from "../api/types";

// Deterministic tone per movie so placeholders are distinguishable but stay
// within the charcoal / burgundy palette.
const TONES = [
  ["#5c1530", "#2c2224"],
  ["#6e2238", "#262022"],
  ["#4a1a2c", "#302628"],
  ["#7a1f3d", "#2a2325"],
  ["#3f2a33", "#221e1f"],
  ["#5a2a2a", "#2b2423"],
];

function Placeholder({ movie, compact }: { movie: Movie; compact: boolean }) {
  const [from, to] = TONES[movie.movieId % TONES.length];
  const genres = movie.genres.length ? movie.genres.slice(0, 3).join(", ") : "No genres listed";
  const label = `Placeholder poster for ${movie.title}${movie.year ? ` (${movie.year})` : ""}: ${genres}`;
  const background = { background: `linear-gradient(160deg, ${from} 0%, ${to} 78%)` };
  if (compact) {
    // Thumbnails sit next to the full title, so show a monogram instead of text.
    const initial = movie.title.replace(/^(The|A|An)\s+/i, "").trim().charAt(0).toUpperCase() || "?";
    return (
      <div role="img" aria-label={label} className="flex h-full w-full items-center justify-center" style={background}>
        <span aria-hidden className="display text-3xl text-paper/85">
          {initial}
        </span>
      </div>
    );
  }
  return (
    <div
      role="img"
      aria-label={label}
      className="relative flex h-full w-full flex-col justify-between overflow-hidden p-3"
      style={background}
    >
      <span aria-hidden className="num text-xs text-paper/70">
        {movie.year ?? "Year unknown"}
      </span>
      <span aria-hidden className="display line-clamp-5 text-xl text-paper [overflow-wrap:anywhere]">
        {movie.title}
      </span>
      <span aria-hidden className="line-clamp-2 text-xs text-paper/75">
        {genres}
      </span>
    </div>
  );
}

export function Poster({ movie, compact = false }: { movie: Movie; compact?: boolean }) {
  const [failed, setFailed] = useState(false);
  return (
    <div className="aspect-[2/3] w-full overflow-hidden rounded-[10px] border border-line/70 bg-raised">
      {movie.posterUrl && !failed ? (
        <img
          src={movie.posterUrl}
          alt={`Poster for ${movie.title}${movie.year ? ` (${movie.year})` : ""}`}
          loading="lazy"
          className="h-full w-full object-cover"
          onError={() => setFailed(true)}
        />
      ) : (
        <Placeholder movie={movie} compact={compact} />
      )}
    </div>
  );
}
