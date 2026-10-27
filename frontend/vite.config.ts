import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The dev server proxies /api to the FastAPI backend, so the browser never
// needs to know the backend address and no secrets reach frontend code.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backend = env.MOVIEMATCH_BACKEND_URL || "http://127.0.0.1:8000";
  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      proxy: { "/api": { target: backend, changeOrigin: true } },
    },
    preview: {
      port: 4173,
      proxy: { "/api": { target: backend, changeOrigin: true } },
    },
  };
});
