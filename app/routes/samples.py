from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

from src.data_loader import load_sensor_csv
from src.sample_datasets import build_all_samples, find_definition

from ..services import _set_dataset
from ..state import get_state

router = APIRouter()


@router.post("/data/samples/load/{slug}")
def load_sample(slug: str):
    state = get_state()
    definition = find_definition(slug)
    if definition is None:
        state.set_flash("error", f"Unknown sample dataset: {slug}")
        return RedirectResponse(url="/data", status_code=303)
    if not definition.path.exists():
        build_all_samples()
    df = load_sensor_csv(definition.path)
    _set_dataset(state, df)
    state.set_flash(
        "success",
        f"Loaded '{definition.title}' — {len(df)} rows, {df['fault_type'].nunique()} classes.",
    )
    return RedirectResponse(url="/data", status_code=303)


@router.get("/data/samples/download/{slug}")
async def download_sample(slug: str):
    state = get_state()
    definition = find_definition(slug)
    if definition is None:
        state.set_flash("error", f"Unknown sample dataset: {slug}")
        return RedirectResponse(url="/data", status_code=303)
    if not definition.path.exists():
        build_all_samples()
    return FileResponse(
        path=definition.path,
        media_type="text/csv",
        filename=definition.file_name,
    )


@router.post("/data/samples/rebuild")
def rebuild_samples():
    state = get_state()
    build_all_samples()
    state.set_flash("success", "All sample datasets rebuilt.")
    return RedirectResponse(url="/data", status_code=303)
