"""Dedicated authenticated FastAPI app; run with --factory and one worker."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sat_rs_vlm.configuration.environment import expand_environment

from .errors import ProtocolError, ProtocolValidationError
from .image_package import ImagePackage
from .models import parse_request
from .question_capture import QuestionCapture
from .service import CompetitionService

logger = logging.getLogger("sat_rs_vlm.competition")


def create_app(
    config_path: str | Path | None = None,
    *,
    service: CompetitionService | None = None,
    api_key: str | None = None,
    timeout_seconds: float | None = None,
    question_capture_path: str | Path | None = None,
) -> FastAPI:
    config_file = Path(
        config_path or os.environ.get("COMPETITION_CONFIG", "configs/competition/protocol_v2.yaml")
    ).resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        secret = api_key or os.environ.get("MODEL_API_KEY")
        if not secret:
            raise RuntimeError("MODEL_API_KEY must be configured")
        app.state.key_digest = hashlib.sha256(secret.encode()).digest()
        config = {}
        active_service = service
        if active_service is None:
            config = expand_environment(
                yaml.safe_load(config_file.read_text(encoding="utf-8")),
                environ=os.environ,
            )
            runtime_path = Path(config["runtime_config"]).expanduser()
            if not runtime_path.is_absolute():
                runtime_path = config_file.parent / runtime_path
            runtime_config = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
            package = ImagePackage(config["package"]["root"])
            active_service = CompetitionService.from_config(
                package,
                runtime_config,
                config,
                offline=config.get("offline", False),
            )
        app.state.service = active_service
        app.state.timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else config.get("server", {}).get("timeout_seconds", 120)
        )
        if not 0 < app.state.timeout <= 120:
            raise ValueError("Competition request timeout must be in (0, 120]")
        app.state.lock = asyncio.Lock()
        app.state.inflight = None
        capture_path = question_capture_path or config.get("logging", {}).get(
            "question_capture_path"
        )
        app.state.question_capture = QuestionCapture(capture_path) if capture_path else None
        try:
            if config.get("server", {}).get("preload", False):
                seen = set()
                for provider in vars(active_service.runtime.providers).values():
                    if id(provider) not in seen and callable(getattr(provider, "preload", None)):
                        await asyncio.to_thread(provider.preload)
                    seen.add(id(provider))
            yield
        finally:
            if app.state.inflight is not None:
                # A timed-out CUDA job remains serialized until it actually finishes.
                await asyncio.gather(app.state.inflight, return_exceptions=True)
            try:
                active_service.close()
            finally:
                if app.state.question_capture is not None:
                    app.state.question_capture.close()

    app = FastAPI(
        title="Competition Protocol 2.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
    )

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        digest = hashlib.sha256(token.encode()).digest()
        if (
            scheme.lower() != "bearer"
            or not token
            or not hmac.compare_digest(digest, request.app.state.key_digest)
        ):
            return JSONResponse(
                {"error": {"code": "unauthorized", "message": "Bearer token required."}},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    @app.exception_handler(ProtocolError)
    async def protocol_error(request: Request, exc: ProtocolError):
        return JSONResponse(
            {"error": {"code": exc.code, "message": exc.public_message}},
            status_code=exc.status_code,
        )

    @app.get("/healthz")
    async def healthz():
        readiness = app.state.service.readiness()
        return {
            "status": readiness["status"],
            "protocol_version": "2.0",
            "dataset_id": app.state.service.package.dataset_id,
            "model_readiness": readiness,
        }

    @app.post("/v1/predict")
    async def predict(request: Request):
        started = time.monotonic()
        if (
            request.headers.get("content-type", "").split(";")[0].strip().lower()
            != "application/json"
        ):
            return JSONResponse(
                {"error": {"code": "content_type", "message": "JSON required."}}, status_code=415
            )
        try:
            parsed = parse_request(await request.json())
        except (ValueError, UnicodeError) as exc:
            raise ProtocolValidationError() from exc
        if app.state.question_capture is not None:
            try:
                app.state.question_capture.record(parsed)
            except OSError:
                # Capture failure must not expose the request or break inference.
                logger.error("question_capture_write_failed")
        if app.state.lock.locked():
            return JSONResponse(
                {"error": {"code": "busy", "message": "Inference is busy."}}, status_code=503
            )
        await app.state.lock.acquire()
        task = asyncio.create_task(asyncio.to_thread(app.state.service.predict, parsed))
        app.state.inflight = task

        def completed(job):
            if not job.cancelled():
                job.exception()  # Consume late failures without logging request contents.
            app.state.inflight = None
            app.state.lock.release()

        task.add_done_callback(completed)
        error_code = None
        try:
            remaining = app.state.timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError()
            response = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            return JSONResponse(response)
        except TimeoutError:
            error_code = "timeout"
            return JSONResponse(
                {"error": {"code": "timeout", "message": "Inference timed out."}}, status_code=504
            )
        except ProtocolError as exc:
            error_code = exc.code
            raise
        except Exception:
            error_code = "runtime_inference_error"
            return JSONResponse(
                {"error": {"code": error_code, "message": "Inference failed."}}, status_code=500
            )
        finally:
            diagnostics = app.state.service.last_diagnostics
            if diagnostics.get("request_id") != parsed.request_id:
                diagnostics = {}
            logger.info(
                json.dumps(
                    {
                        "request_id": parsed.request_id,
                        "item_id": parsed.item_id,
                        "constraint": parsed.response_constraint.type,
                        "image_count": len(parsed.images),
                        "execution_mode": diagnostics.get("execution_mode"),
                        "dimensions": diagnostics.get("dimensions"),
                        "latency_ms": round((time.monotonic() - started) * 1000, 2),
                        "error_code": error_code,
                    },
                    ensure_ascii=True,
                )
            )

    return app
