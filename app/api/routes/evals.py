"""The retrieval benchmark an admin runs: start it, follow it, list past runs.

The gateway authenticates the admin. A run takes minutes, so POST answers at once with its
id and the page polls GET until the status is no longer queued or running.
"""

import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path

from app.schemas.evals import EvalOptions, EvalRequest, EvalRun, EvalRunList
from evals.runners import admin_run

router = APIRouter(prefix="/admin/evals", tags=["admin"])


@router.get("/options", response_model=EvalOptions)
def eval_options() -> EvalOptions:
    return admin_run.options()


@router.post("/retrieval", response_model=EvalRun, status_code=202)
async def start_run(body: EvalRequest) -> EvalRun:
    try:
        models = admin_run.choose_models(body)
        # In a thread: it needs its own event loop (see run_in_selector_loop).
        await asyncio.to_thread(
            lambda: admin_run.run_in_selector_loop(
                admin_run.check_vectors(body, models.embedding)
            )
        )
    except admin_run.Rejected as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from None
    return admin_run.submit(uuid.uuid4().hex, body, models, asyncio.get_running_loop())


@router.get("/retrieval", response_model=EvalRunList)
def list_runs() -> EvalRunList:
    return EvalRunList(runs=admin_run.store.list())


@router.get("/retrieval/{run_id}", response_model=EvalRun)
def get_run(run_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]) -> EvalRun:
    run = admin_run.store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy lần chạy.")
    return run
