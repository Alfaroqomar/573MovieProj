import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import { Layout } from "./components/Layout";
import { StatusProvider } from "./lib/status";
import { AboutPage } from "./pages/AboutPage";
import { ComparePage } from "./pages/ComparePage";
import { HomePage } from "./pages/HomePage";
import { RecommendationsPage } from "./pages/RecommendationsPage";

function NotFound() {
  return (
    <div className="measure space-y-2">
      <h1 className="display text-3xl text-paper">Page not found</h1>
      <p className="text-muted">MovieMatch has four pages: Home, Recommendations, Compare models, and About.</p>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <StatusProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<HomePage />} />
            <Route path="recommendations" element={<RecommendationsPage />} />
            <Route path="compare" element={<ComparePage />} />
            <Route path="about" element={<AboutPage />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </StatusProvider>
  </StrictMode>,
);
