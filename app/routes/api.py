from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import services
from ..state import get_state

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/health")
async def health() -> dict:
    state = get_state()
    return {
        "status": "ok",
        "dataset_loaded": state.raw_df is not None,
        "models_trained": list(state.trained_models.keys()),
        "best_model": state.best_model_name,
    }


@router.get("/dataset/summary")
async def dataset_summary() -> dict:
    state = get_state()
    if state.raw_df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")
    df = state.raw_df
    fault_counts = (
        df["fault_type"].value_counts().to_dict() if "fault_type" in df.columns else {}
    )
    return {
        "rows": len(df),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "fault_counts": fault_counts,
    }


@router.post("/dataset/generate")
def api_generate(samples_per_class: int = 300, seed: int = 42) -> dict:
    state = get_state()
    samples_per_class = max(50, min(samples_per_class, 1500))
    df = services.generate_synthetic(state, samples_per_class=samples_per_class, seed=seed)
    return {"rows": len(df), "classes": int(df["fault_type"].nunique())}


@router.post("/train")
def api_train() -> dict:
    state = get_state()
    try:
        outcome = services.train_models(state)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    metrics = outcome["metrics"]
    return {
        "best_model": outcome["best_model"],
        "n_train": outcome["n_train"],
        "n_test": outcome["n_test"],
        "metrics": {
            "accuracy": metrics.accuracy,
            "precision_macro": metrics.precision,
            "recall_macro": metrics.recall,
            "f1_macro": metrics.f1,
        },
        "all_models": {
            name: {
                "accuracy": r.accuracy,
                "f1_macro": r.f1,
                "train_seconds": r.train_seconds,
            }
            for name, r in state.evaluation_results.items()
        },
    }


@router.post("/predict")
def api_predict(model_name: str | None = None) -> dict:
    state = get_state()
    try:
        records = services.predict_on_current_dataset(state, model_name=model_name)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "rows": len(records),
        "model": model_name or state.best_model_name,
        "fault_counts": records["predicted_fault"].value_counts().to_dict(),
        "average_confidence": float(records["confidence"].mean())
        if records["confidence"].notna().any()
        else None,
    }
