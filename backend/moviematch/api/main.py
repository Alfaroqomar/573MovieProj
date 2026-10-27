"""FastAPI application.

Start with:  uvicorn moviematch.api.main:app --port 8000
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from moviematch.api.schemas import (
    AlsRecommendationRequest,
    LlmRecommendationRequest,
    RecommendationRequest,
    RecommendationResponse,
)
from moviematch.config import Settings, get_settings
from moviematch.data.raw import DATASET_PAGE, DATASET_README, DATASET_ZIP
from moviematch.llm.reranker import ChatClient
from moviematch.service import ApiError, MovieMatchService

logger = logging.getLogger("moviematch.api")


def create_app(settings: Settings | None = None, llm_client: ChatClient | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = MovieMatchService(settings, llm_client=llm_client)
        service.load()
        state = service.readiness()
        logger.info("MovieMatch backend state: %s", state.state)
        for message in state.messages:
            logger.info("%s: %s %s", message["code"], message["message"], message["action"])
        app.state.service = service
        yield

    app = FastAPI(title="MovieMatch API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def service(request: Request) -> MovieMatchService:
        return request.app.state.service

    # ------------------------------ errors --------------------------------- #

    def error_body(code: str, message: str) -> dict[str, Any]:
        return {"error": {"code": code, "message": message}}

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", []) if p not in ("body", "query"))
        message = f"Invalid {field}: {first.get('msg', 'invalid value')}" if field else "Invalid request."
        return JSONResponse(error_body("invalid_request", message), status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")  # details stay in the server log
        return JSONResponse(
            error_body("internal_error", "An unexpected server error occurred."), status_code=500
        )

    # ------------------------------ status --------------------------------- #

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "state": service(request).readiness().state}

    @app.get("/api/config/status")
    def config_status(request: Request) -> dict[str, Any]:
        svc = service(request)
        state = svc.readiness()
        modes = {
            "popularity": state.models["popularity"],
            "item_knn": state.models["item_knn"],
            "als": state.models["als"],
            "als_llm": state.models["als_llm"],
        }
        default_mode = "als_llm" if modes["als_llm"] else "als"
        return {
            **state.as_dict(),
            "modes": modes,
            "defaultMode": default_mode,
            "posterSourceConfigured": bool(svc.settings.poster_url_template),
            "loadErrors": svc.load_errors,
        }

    @app.get("/api/dataset/status")
    def dataset_status(request: Request) -> dict[str, Any]:
        svc = service(request)
        state = svc.readiness()
        manifest = getattr(svc, "manifest", None) if svc.loaded["processed"] else None
        return {
            "state": state.state,
            "rawDataset": state.raw_dataset,
            "processed": state.processed,
            "processedProblem": state.processed_problem,
            "loaded": svc.loaded["processed"],
            "dataset": manifest["dataset"] if manifest else None,
            "preprocessing": manifest["config"] if manifest else None,
            "processedAt": manifest["createdAt"] if manifest else None,
            "links": {"page": DATASET_PAGE, "readme": DATASET_README, "zip": DATASET_ZIP},
            "messages": [m for m in state.messages if m["code"] != "llm_not_configured"],
        }

    @app.get("/api/dataset/statistics")
    def dataset_statistics(request: Request) -> dict[str, Any]:
        return service(request).statistics_out()

    # ------------------------------ lookups -------------------------------- #

    @app.get("/api/users/search")
    def users_search(
        request: Request,
        query: str = Query(default="", max_length=12),
        page: int = Query(default=1, ge=1),
        pageSize: int = Query(default=20, ge=1, le=50),
    ) -> dict[str, Any]:
        return service(request).search_users(query, page, pageSize)

    @app.get("/api/users/{user_id}/summary")
    def user_summary(request: Request, user_id: int) -> dict[str, Any]:
        return service(request).user_summary(user_id)

    @app.get("/api/movies/{movie_id}")
    def movie(request: Request, movie_id: int) -> dict[str, Any]:
        return service(request).movie_details(movie_id)

    # -------------------------- recommendations ---------------------------- #

    def respond(events: Iterator[dict[str, Any]], stream: bool):
        """Return the final JSON result, or stream NDJSON progress events.

        Request problems (unknown user, missing model, LLM not configured)
        are raised by ``service.run`` before this point and become normal
        HTTP errors. Errors after streaming has begun are sent as an
        ``{"type": "error"}`` line.
        """
        if not stream:
            for event in events:
                if event["type"] == "result":
                    return RecommendationResponse.model_validate(event["data"]).model_dump()
            raise ApiError(500, "internal_error", "No result was produced.")

        def lines() -> Iterator[str]:
            try:
                for event in events:
                    if event["type"] == "result":
                        data = RecommendationResponse.model_validate(event["data"]).model_dump()
                        event = {"type": "result", "data": data}
                    yield json.dumps(event) + "\n"
            except ApiError as exc:
                yield json.dumps({"type": "error", **error_body(exc.code, exc.message)}) + "\n"
            except Exception:  # noqa: BLE001
                logger.exception("Unhandled error while streaming recommendations")
                body = error_body("internal_error", "An unexpected server error occurred.")
                yield json.dumps({"type": "error", **body}) + "\n"

        return StreamingResponse(
            lines(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/recommendations/popularity")
    def recommend_popularity(request: Request, body: RecommendationRequest, stream: bool = False):
        return respond(service(request).run("popularity", body.userId, body.topN), stream)

    @app.post("/api/recommendations/knn")
    def recommend_knn(request: Request, body: RecommendationRequest, stream: bool = False):
        return respond(service(request).run("item_knn", body.userId, body.topN), stream)

    @app.post("/api/recommendations/als")
    def recommend_als(request: Request, body: AlsRecommendationRequest, stream: bool = False):
        return respond(
            service(request).run("als", body.userId, body.topN, body.candidateCount), stream
        )

    @app.post("/api/recommendations/als-llm")
    def recommend_als_llm(request: Request, body: LlmRecommendationRequest, stream: bool = False):
        return respond(
            service(request).run(
                "als_llm", body.userId, body.topN, body.candidateCount, body.userRequest
            ),
            stream,
        )

    # ----------------------------- evaluation ------------------------------ #

    @app.get("/api/evaluation/results")
    def evaluation_results(request: Request) -> dict[str, Any]:
        return service(request).evaluation_results()

    return app


app = create_app()
