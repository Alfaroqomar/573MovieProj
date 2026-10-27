import type { ButtonHTMLAttributes, ReactNode } from "react";

export function GenreList({ genres, small = false }: { genres: string[]; small?: boolean }) {
  if (!genres.length) return <p className="text-sm text-muted">No genres listed</p>;
  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Genres">
      {genres.map((g) => (
        <li
          key={g}
          className={`rounded-full border border-line px-2 ${small ? "py-0 text-xs" : "py-0.5 text-[0.8rem]"} text-muted`}
        >
          {g}
        </li>
      ))}
    </ul>
  );
}

export function ModeBadge({ label, color }: { label: string; color: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted">
      <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}

/** Label/value rows for model scores. */
export function Facts({ rows }: { rows: Array<[string, ReactNode] | null | false> }) {
  const shown = rows.filter(Boolean) as Array<[string, ReactNode]>;
  return (
    <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 text-sm">
      {shown.map(([label, value]) => (
        <div key={label} className="contents">
          <dt className="text-muted">{label}</dt>
          <dd className="num text-right text-paper">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Button({
  children,
  variant = "primary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" }) {
  const styles = {
    primary: "bg-wine text-paper hover:bg-[#8e2648] border border-wine disabled:bg-raised disabled:border-line disabled:text-faint",
    secondary: "border border-line bg-raised text-paper hover:border-muted disabled:text-faint",
    ghost: "text-muted hover:text-paper hover:bg-raised border border-transparent",
  }[variant];
  return (
    <button
      type="button"
      className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-[var(--radius-control)] px-4 text-sm font-semibold transition-colors disabled:cursor-not-allowed ${styles} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function SectionHeading({ id, children, sub }: { id?: string; children: ReactNode; sub?: ReactNode }) {
  return (
    <div className="mb-5">
      <h2 id={id} className="display-soft text-2xl text-paper sm:text-[1.7rem]">
        {children}
      </h2>
      {sub ? <p className="measure mt-1.5 text-muted">{sub}</p> : null}
    </div>
  );
}

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-[var(--radius-card)] border border-line bg-panel shadow-[var(--shadow-card)] ${className}`}>
      {children}
    </div>
  );
}
