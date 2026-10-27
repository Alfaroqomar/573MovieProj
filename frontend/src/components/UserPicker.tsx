import { useEffect, useId, useState } from "react";
import { ApiError, api } from "../api/client";
import type { UserSearchResult, UserSummary } from "../api/types";
import { fmtDate, fmtInt } from "../lib/modes";
import { Alert } from "./Alert";

const PAGE_SIZE = 8;

/** Server-side, paginated search over MovieLens user IDs. */
export function UserPicker({
  selectedUserId,
  onSelect,
  disabled,
}: {
  selectedUserId: number | null;
  onSelect: (userId: number) => void;
  disabled?: boolean;
}) {
  const inputId = useId();
  const helpId = useId();
  const [query, setQuery] = useState(selectedUserId ? String(selectedUserId) : "");
  const [page, setPage] = useState(1);
  const [result, setResult] = useState<UserSearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  const invalid = query.trim() !== "" && !/^\d+$/.test(query.trim());

  useEffect(() => {
    if (!open || disabled || invalid) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      api
        .searchUsers(query.trim(), page, PAGE_SIZE)
        .then((r) => {
          if (!cancelled) {
            setResult(r);
            setError(null);
          }
        })
        .catch((e: ApiError) => !cancelled && setError(e.message));
    }, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [query, page, open, disabled, invalid]);

  const choose = (userId: number) => {
    setQuery(String(userId));
    setOpen(false);
    onSelect(userId);
  };

  return (
    <div>
      <label htmlFor={inputId} className="mb-1.5 block text-sm font-semibold text-paper">
        MovieLens user ID
      </label>
      <div className="flex gap-2">
        <input
          id={inputId}
          type="text"
          inputMode="numeric"
          autoComplete="off"
          placeholder="Search, e.g. 42"
          aria-describedby={helpId}
          aria-invalid={invalid}
          disabled={disabled}
          value={query}
          onFocus={() => setOpen(true)}
          onChange={(e) => {
            setQuery(e.target.value);
            setPage(1);
            setOpen(true);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && /^\d+$/.test(query.trim())) {
              e.preventDefault();
              choose(Number(query.trim()));
            }
            if (e.key === "Escape") setOpen(false);
          }}
          className="min-h-11 w-full rounded-[var(--radius-control)] border border-line bg-raised px-3 text-paper placeholder:text-faint disabled:opacity-60"
        />
      </div>
      <p id={helpId} className="mt-1 text-xs text-muted">
        {invalid ? "User IDs contain digits only." : "Matches IDs that start with what you type. Press Enter to use an exact ID."}
      </p>

      {open && !invalid && !disabled ? (
        <div className="mt-2 rounded-[var(--radius-control)] border border-line bg-ink/60 p-1.5">
          {error ? <Alert tone="error" title="User search failed">{error}</Alert> : null}
          {result && !error ? (
            <>
              <p className="px-2 py-1 text-xs text-muted" aria-live="polite">
                {result.total === 0
                  ? "No MovieLens users match."
                  : `${fmtInt(result.total)} matching users, page ${result.page} of ${fmtInt(result.totalPages)}`}
              </p>
              <ul aria-label="Matching users">
                {result.users.map((u) => (
                  <li key={u.userId}>
                    <button
                      type="button"
                      onClick={() => choose(u.userId)}
                      className={`flex w-full items-center justify-between gap-3 rounded-lg px-2 py-2 text-left text-sm hover:bg-raised ${
                        u.userId === selectedUserId ? "bg-raised" : ""
                      }`}
                    >
                      <span className="num font-semibold text-paper">User {u.userId}</span>
                      <span className="num truncate text-xs text-muted">
                        {fmtInt(u.ratingCount)} ratings
                        {u.favoriteGenres.length ? `, ${u.favoriteGenres.slice(0, 2).join(", ")}` : ""}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
              {result.totalPages > 1 ? (
                <div className="flex justify-between px-1 pt-1">
                  <button
                    type="button"
                    className="min-h-9 rounded-lg px-2 text-sm text-muted hover:text-paper disabled:opacity-40"
                    disabled={page <= 1}
                    onClick={() => setPage((p) => p - 1)}
                  >
                    Previous page
                  </button>
                  <button
                    type="button"
                    className="min-h-9 rounded-lg px-2 text-sm text-muted hover:text-paper disabled:opacity-40"
                    disabled={page >= result.totalPages}
                    onClick={() => setPage((p) => p + 1)}
                  >
                    Next page
                  </button>
                </div>
              ) : null}
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/** Loads a user's summary from the backend whenever the ID changes. */
export function useUserSummary(userId: number | null) {
  const [summary, setSummary] = useState<UserSummary | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (userId == null) {
      setSummary(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    api
      .userSummary(userId)
      .then((s) => {
        if (!cancelled) {
          setSummary(s);
          setError(null);
        }
      })
      .catch((e: ApiError) => {
        if (!cancelled) {
          setSummary(null);
          setError(e);
        }
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [userId]);

  return { summary, error, loading };
}

export function UserSummaryCard({ summary, compact = false }: { summary: UserSummary; compact?: boolean }) {
  return (
    <section aria-label={`Summary for user ${summary.userId}`} className="space-y-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="display-soft text-lg text-paper">User {summary.userId}</h3>
        <span className="num text-sm text-muted">{fmtInt(summary.ratingCount)} ratings</span>
      </div>
      <p className="text-sm text-muted">
        Rated between {fmtDate(summary.firstRatingAt)} and {fmtDate(summary.lastRatingAt)}, averaging{" "}
        <span className="num">{summary.meanRating.toFixed(2)}</span>.
      </p>
      <div>
        <p className="text-sm text-muted">Favorite genres</p>
        <p className="text-sm text-paper">{summary.favoriteGenres.join(", ") || "None recorded"}</p>
      </div>
      <div>
        <p className="text-sm text-muted">Highly rated ({summary.highRatingThreshold.toFixed(1)} or higher)</p>
        {summary.highlyRatedMovies.length ? (
          <ul className="mt-1 space-y-0.5 text-sm">
            {summary.highlyRatedMovies.slice(0, compact ? 4 : 6).map((m) => (
              <li key={m.movieId} className="flex justify-between gap-2">
                <span className="truncate text-paper">
                  {m.title}
                  {m.year ? <span className="text-muted"> ({m.year})</span> : null}
                </span>
                <span className="num shrink-0 text-muted">{m.rating.toFixed(1)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-paper">This user has no ratings at or above the threshold.</p>
        )}
      </div>
    </section>
  );
}
