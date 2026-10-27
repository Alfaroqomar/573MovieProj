import type { ReactNode } from "react";

type Tone = "error" | "warning" | "info" | "success";

const TONES: Record<Tone, { border: string; dot: string; label: string }> = {
  error: { border: "border-bad/50", dot: "bg-bad", label: "Error" },
  warning: { border: "border-warn/50", dot: "bg-warn", label: "Warning" },
  info: { border: "border-line", dot: "bg-muted", label: "Note" },
  success: { border: "border-ok/50", dot: "bg-ok", label: "Done" },
};

export function Alert({
  tone,
  title,
  children,
  live = false,
}: {
  tone: Tone;
  title: ReactNode;
  children?: ReactNode;
  live?: boolean;
}) {
  const t = TONES[tone];
  return (
    <div
      role={live ? (tone === "error" ? "alert" : "status") : undefined}
      className={`flex gap-3 rounded-[var(--radius-control)] border ${t.border} bg-panel px-4 py-3`}
    >
      <span aria-hidden className={`mt-2 h-2 w-2 shrink-0 rounded-full ${t.dot}`} />
      <div className="min-w-0 text-sm">
        <p className="font-semibold text-paper">
          <span className="sr-only">{t.label}: </span>
          {title}
        </p>
        {children ? <div className="mt-1 text-muted">{children}</div> : null}
      </div>
    </div>
  );
}

export function Code({ children }: { children: ReactNode }) {
  return (
    <code className="rounded-md bg-raised px-1.5 py-0.5 text-[0.85em] text-paper [overflow-wrap:anywhere]">
      {children}
    </code>
  );
}
