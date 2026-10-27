import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { ApiError, api } from "../api/client";
import type { ConfigStatus } from "../api/types";

interface StatusContextValue {
  status: ConfigStatus | null;
  error: ApiError | null;
  loading: boolean;
  refresh: () => void;
}

const StatusContext = createContext<StatusContextValue | null>(null);

export function StatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ConfigStatus | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    api
      .configStatus()
      .then((s) => {
        setStatus(s);
        setError(null);
      })
      .catch((e: ApiError) => {
        setStatus(null);
        setError(e);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(refresh, [refresh]);

  return <StatusContext.Provider value={{ status, error, loading, refresh }}>{children}</StatusContext.Provider>;
}

export function useStatus(): StatusContextValue {
  const value = useContext(StatusContext);
  if (!value) throw new Error("useStatus must be used inside StatusProvider");
  return value;
}

/** True when the processed dataset is loaded, so lookups (users, statistics) work. */
export function datasetReady(status: ConfigStatus | null): boolean {
  return !!status && status.processed && status.state !== "dataset_unavailable" && status.state !== "not_processed";
}
