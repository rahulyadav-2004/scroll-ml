import os
import time
from threading import Event, Lock, RLock, Thread

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

try:
    import orjson
    from fastapi.responses import ORJSONResponse

    RESPONSE_CLASS = ORJSONResponse
except ImportError:
    from fastapi.responses import JSONResponse

    RESPONSE_CLASS = JSONResponse

try:
    from .feature_engineering import CORE_FEATURE_COLUMNS, FEATURE_COLUMNS, ensure_feature_frame
    from .training_pipeline import (
        AutoTrainingConfig,
        read_active_manifest,
        read_training_status,
        run_training_cycle,
    )
except ImportError:
    from feature_engineering import CORE_FEATURE_COLUMNS, FEATURE_COLUMNS, ensure_feature_frame
    from training_pipeline import (
        AutoTrainingConfig,
        read_active_manifest,
        read_training_status,
        run_training_cycle,
    )


load_dotenv()

app = FastAPI(
    title="Scroll ML Inference Service",
    description="Batch scoring for Shadow Mode ranking.",
    version="1.2.0",
    default_response_class=RESPONSE_CLASS,
)

MODEL_PATH = os.path.join("models", "v1_shadow.joblib")
COMMERCE_MODEL_PATH = os.path.join("models", "commerce_v1.joblib")


def _utc_now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_model(path):
    if not os.path.exists(path):
        print(f"⚠️ Model not found at {path}. Pipeline degradation active.")
        return None
    try:
        print(f"📦 Loading model from {path}...")
        model = joblib.load(path)
        print(f"✅ Model {path} loaded successfully.")
        return model
    except Exception as exc:
        print(f"❌ Failed to load model {path}: {exc}")
        return None


class ModelRegistry:
    def __init__(self):
        self._lock = RLock()
        self.engagement = None
        self.commerce = None
        self.active_manifest = {}
        self.last_loaded_at = None
        self.reload_models()

    def reload_models(self):
        engagement = load_model(MODEL_PATH)
        commerce = load_model(COMMERCE_MODEL_PATH)
        manifest = read_active_manifest() or {}

        with self._lock:
            self.engagement = engagement
            self.commerce = commerce
            self.active_manifest = manifest
            self.last_loaded_at = _utc_now_iso()

        return self.snapshot()

    def get_models(self):
        with self._lock:
            return self.engagement, self.commerce

    def snapshot(self):
        with self._lock:
            return {
                "engagement_loaded": self.engagement is not None,
                "commerce_loaded": self.commerce is not None,
                "active_manifest": self.active_manifest,
                "last_loaded_at": self.last_loaded_at,
            }


model_registry = ModelRegistry()
training_config = AutoTrainingConfig.from_env()
training_lock = Lock()
training_stop_event = Event()
auto_training_thread = None


def run_training_job(trigger="manual", force_promote_fallback=False):
    if not training_lock.acquire(blocking=False):
        return {
            "started": False,
            "reason": "training_already_running",
            "status": read_training_status(),
        }

    try:
        result = run_training_cycle(
            config=training_config,
            force_promote_fallback=force_promote_fallback,
            trigger=trigger,
        )
        if result.get("promoted"):
            model_registry.reload_models()
        return {
            "started": True,
            "result": result,
        }
    finally:
        training_lock.release()


def start_training_in_background(trigger="manual", force_promote_fallback=False):
    if training_lock.locked():
        return False

    def _runner():
        run_training_job(
            trigger=trigger,
            force_promote_fallback=force_promote_fallback,
        )

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    return True


def auto_training_loop():
    if training_config.initial_delay_seconds > 0:
        if training_stop_event.wait(training_config.initial_delay_seconds):
            return

    while not training_stop_event.is_set():
        run_training_job(trigger="scheduled")
        if training_stop_event.wait(training_config.interval_seconds):
            break


@app.on_event("startup")
def on_startup():
    global auto_training_thread
    if training_config.enabled and auto_training_thread is None:
        auto_training_thread = Thread(target=auto_training_loop, daemon=True)
        auto_training_thread.start()


@app.on_event("shutdown")
def on_shutdown():
    training_stop_event.set()


class ScrollFeature(BaseModel):
    position_index: int
    user_category_score: float
    video_quality: float
    has_products: int
    hour_of_day: int
    completion_rate: float
    expected_ctr_at_position: float = 0.0
    session_velocity: float = 1.0
    session_dwell_time: float = 0.0
    is_product: int = 0
    creator_affinity_score: float = 0.0
    content_freshness: float = 0.0
    global_ctr: float = 0.0
    global_conversion_rate: float = 0.0
    social_proof_score: float = 0.0
    semantic_score: float = 0.0
    retrieval_score: float = 0.0
    has_semantic_candidate: int = 0
    semantic_profile_strength: float = 0.0


class BatchRequest(BaseModel):
    items: list[ScrollFeature]


def resolve_model_feature_columns(model):
    if model is None:
        return CORE_FEATURE_COLUMNS

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return CORE_FEATURE_COLUMNS

    return [column for column in feature_names if column in FEATURE_COLUMNS] or CORE_FEATURE_COLUMNS


@app.get("/health")
def health():
    model_snapshot = model_registry.snapshot()
    return {
        "status": "healthy",
        "models": {
            "engagement": "v1_shadow" if model_snapshot["engagement_loaded"] else "missing",
            "commerce": "commerce_v1" if model_snapshot["commerce_loaded"] else "missing",
        },
        "training": {
            "auto_enabled": training_config.enabled,
            "status": read_training_status(),
            "active_manifest": model_snapshot["active_manifest"],
            "last_loaded_at": model_snapshot["last_loaded_at"],
        },
    }


@app.post("/predict")
def predict(batch: BatchRequest):
    """
    Multi-task inference for engagement and commerce ranking.
    """
    start_time = time.time()
    model_engagement, model_commerce = model_registry.get_models()

    try:
        raw_frame = pd.DataFrame(
            [
                {
                    "position_index": item.position_index,
                    "user_category_score": item.user_category_score,
                    "video_quality": item.video_quality,
                    "has_products": item.has_products,
                    "hour_of_day": item.hour_of_day,
                    "completion_rate": item.completion_rate,
                    "expected_ctr_at_position": item.expected_ctr_at_position,
                    "session_velocity": item.session_velocity,
                    "session_dwell_time": item.session_dwell_time,
                    "is_product": item.is_product,
                    "creator_affinity_score": item.creator_affinity_score,
                    "content_freshness": item.content_freshness,
                    "global_ctr": item.global_ctr,
                    "global_conversion_rate": item.global_conversion_rate,
                    "social_proof_score": item.social_proof_score,
                    "semantic_score": item.semantic_score,
                    "retrieval_score": item.retrieval_score,
                    "has_semantic_candidate": item.has_semantic_candidate,
                    "semantic_profile_strength": item.semantic_profile_strength,
                }
                for item in batch.items
            ]
        )

        X_df = ensure_feature_frame(raw_frame)
        engagement_feature_columns = resolve_model_feature_columns(model_engagement)
        commerce_feature_columns = resolve_model_feature_columns(model_commerce)

        results = {}

        if model_engagement:
            results["engagement_scores"] = model_engagement.predict_proba(
                X_df[engagement_feature_columns]
            )[:, 1].tolist()

        if model_commerce:
            results["commerce_scores"] = model_commerce.predict_proba(
                X_df[commerce_feature_columns]
            )[:, 1].tolist()

        engagement_scores = results.get("engagement_scores", [0.5] * len(batch.items))
        commerce_scores = results.get("commerce_scores", [0.5] * len(batch.items))

        completion_scores = []
        save_scores = []
        product_click_scores = []
        fast_skip_scores = []
        final_rank_scores = []

        for idx, item in enumerate(batch.items):
            engagement = float(engagement_scores[idx])
            commerce = float(commerce_scores[idx])
            completion_rate = float(item.completion_rate)
            freshness_hint = max(0.0, min(1.0, item.expected_ctr_at_position))

            qualified_view = max(0.0, min(1.0, engagement))
            completion_score = max(0.0, min(1.0, (0.65 * engagement) + (0.35 * completion_rate)))
            save_score = max(0.0, min(1.0, (0.55 * engagement) + (0.45 * commerce)))
            product_click_score = max(0.0, min(1.0, commerce))
            fast_skip_score = max(
                0.0,
                min(1.0, 1.0 - ((0.7 * engagement) + (0.3 * completion_rate))),
            )
            final_rank_score = max(
                0.0,
                min(
                    1.0,
                    (0.35 * qualified_view)
                    + (0.20 * completion_score)
                    + (0.15 * save_score)
                    + (0.15 * product_click_score)
                    + (0.10 * freshness_hint)
                    - (0.20 * fast_skip_score),
                ),
            )

            completion_scores.append(completion_score)
            save_scores.append(save_score)
            product_click_scores.append(product_click_score)
            fast_skip_scores.append(fast_skip_score)
            final_rank_scores.append(final_rank_score)

        results["qualified_view_scores"] = engagement_scores
        results["completion_scores"] = completion_scores
        results["save_scores"] = save_scores
        results["product_click_scores"] = product_click_scores
        results["fast_skip_scores"] = fast_skip_scores
        results["final_rank_scores"] = final_rank_scores

        latency_ms = (time.time() - start_time) * 1000

        return {
            **results,
            "ranker_version": "phase7-auto-train-hybrid-v1",
            "latency_ms": round(latency_ms, 2),
            "count": len(batch.items),
            "model_loaded_at": model_registry.snapshot()["last_loaded_at"],
        }
    except Exception as exc:
        print(f"❌ Multi-task inference error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/training/status")
def training_status():
    return {
        "config": {
            "enabled": training_config.enabled,
            "interval_seconds": training_config.interval_seconds,
            "initial_delay_seconds": training_config.initial_delay_seconds,
            "dataset_limit": training_config.dataset_limit,
            "min_rows": training_config.min_rows,
            "min_click_positives": training_config.min_click_positives,
            "min_commerce_positives": training_config.min_commerce_positives,
            "allow_synthetic_fallback_promotion": training_config.allow_synthetic_fallback_promotion,
        },
        "status": read_training_status(),
        "active_manifest": model_registry.snapshot()["active_manifest"],
    }


@app.post("/admin/retrain")
def admin_retrain(
    force_promote_fallback: bool = False,
    x_ml_admin_token: str | None = Header(default=None),
):
    expected_token = os.getenv("ML_ADMIN_TOKEN")
    if expected_token and x_ml_admin_token != expected_token:
        raise HTTPException(status_code=403, detail="invalid admin token")

    started = start_training_in_background(
        trigger="manual",
        force_promote_fallback=force_promote_fallback,
    )
    if not started:
        return {
            "accepted": False,
            "reason": "training_already_running",
            "status": read_training_status(),
        }

    return {
        "accepted": True,
        "message": "training job started",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
