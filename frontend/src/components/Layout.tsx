import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useStatus } from "../lib/status";
import { Alert } from "./Alert";

const LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/recommendations", label: "Recommendations" },
  { to: "/compare", label: "Compare models" },
  { to: "/about", label: "About" },
];

function Nav() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `block rounded-lg px-3 py-2.5 text-[0.95rem] transition-colors sm:py-2 ${
      isActive ? "bg-raised text-paper" : "text-muted hover:bg-raised/60 hover:text-paper"
    }`;

  return (
    <header className="border-b border-line bg-ink/95">
      <nav aria-label="Main" className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 px-4 py-3 sm:px-6">
        <NavLink to="/" className="display mr-auto py-1 text-xl tracking-tight text-paper" end>
          Movie<span className="text-rose">Match</span>
        </NavLink>
        <button
          type="button"
          className="rounded-lg border border-line px-3 py-2 text-sm text-paper sm:hidden"
          aria-expanded={open}
          aria-controls="main-menu"
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "Close" : "Menu"}
        </button>
        <ul
          id="main-menu"
          className={`${open ? "flex" : "hidden"} w-full flex-col gap-1 pt-3 sm:flex sm:w-auto sm:flex-row sm:pt-0`}
        >
          {LINKS.map((link) => (
            <li key={link.to}>
              <NavLink to={link.to} end={link.end} className={linkClass}>
                {link.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}

function StatusBanner() {
  const { status, error, loading } = useStatus();
  if (loading && !status) return null;
  if (error) {
    return (
      <Alert tone="error" title="Backend unavailable" live>
        {error.message}
      </Alert>
    );
  }
  const serious = (status?.messages ?? []).filter((m) => m.level !== "info");
  if (!serious.length) return null;
  return (
    <div className="space-y-2">
      {serious.map((m) => (
        <Alert key={m.code} tone={m.level === "error" ? "error" : "warning"} title={m.message}>
          {m.action}
        </Alert>
      ))}
    </div>
  );
}

export function Layout() {
  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-wine focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <Nav />
      <div className="mx-auto w-full max-w-7xl px-4 pt-4 sm:px-6 empty:hidden">
        <StatusBanner />
      </div>
      <main id="main" className="mx-auto w-full max-w-7xl flex-1 px-4 pb-16 pt-6 sm:px-6">
        <Outlet />
      </main>
      <footer className="border-t border-line">
        <div className="mx-auto max-w-7xl px-4 py-6 text-sm text-muted sm:px-6">
          Data: MovieLens 32M, F. Maxwell Harper and Joseph A. Konstan, GroupLens Research. Recommendations are computed
          locally from the dataset; nothing on this site is sample data.
        </div>
      </footer>
    </div>
  );
}
