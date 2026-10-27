import type { RecommendationMode, StageName } from "../api/types";
import { STAGES_BY_MODE, STAGE_LABELS } from "../lib/modes";

export type StageState = "pending" | "active" | "done" | "failed";

/** Stage checklist driven by real progress events streamed from the backend. */
export function ProgressSteps({ mode, states }: { mode: RecommendationMode; states: Partial<Record<StageName, StageState>> }) {
  const stages = STAGES_BY_MODE[mode];
  const active = stages.find((s) => states[s] === "active");
  return (
    <div className="rounded-[var(--radius-card)] border border-line bg-panel p-5">
      <p className="sr-only" role="status" aria-live="polite">
        {active ? `${STAGE_LABELS[active]}…` : ""}
      </p>
      <ol className="space-y-2.5" aria-label="Progress">
        {stages.map((stage) => {
          const state = states[stage] ?? "pending";
          return (
            <li key={stage} className="flex items-center gap-3 text-sm">
              <span
                aria-hidden
                className={`flex h-5 w-5 items-center justify-center rounded-full border text-[0.65rem] ${
                  state === "done"
                    ? "border-ok bg-ok/15 text-ok"
                    : state === "failed"
                      ? "border-bad bg-bad/15 text-bad"
                      : state === "active"
                        ? "border-rose motion-safe:animate-pulse"
                        : "border-line"
                }`}
              >
                {state === "done" ? "✓" : state === "failed" ? "!" : ""}
              </span>
              <span className={state === "pending" ? "text-faint" : "text-paper"}>
                {STAGE_LABELS[stage]}
                <span className="sr-only">
                  {state === "done" ? ", done" : state === "failed" ? ", failed" : state === "active" ? ", in progress" : ", waiting"}
                </span>
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
