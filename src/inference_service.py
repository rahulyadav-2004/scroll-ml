from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import time
try:
    import orjson
    from fastapi.responses import ORJSONResponse
    RESPONSE_CLASS = ORJSONResponse
except ImportError:
    from fastapi.responses import JSONResponse
    RESPONSE_CLASS = JSONResponse

try:
    from .feature_engineering import FEATURE_COLUMNS, ensure_feature_frame
except ImportError:
    from feature_engineering import FEATURE_COLUMNS, ensure_feature_frame

app = FastAPI(
    title="Scroll ML Inference Service",
    description="Batch scoring for Shadow Mode ranking.",
    version="1.1.0",
    default_response_class=RESPONSE_CLASS
)

# Load the trained models
MODEL_PATH = os.path.join("models", "v1_shadow.joblib")
COMMERCE_MODEL_PATH = os.path.join("models", "commerce_v1.joblib")

def load_model(path):
    if not os.path.exists(path):
        print(f"⚠️ Model not found at {path}. Pipeline degradation active.")
        return None
    try:
        print(f"📦 Loading model from {path}...")
        model = joblib.load(path)
        print(f"✅ Model {path} loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Failed to load model {path}: {e}")
        return None

model_engagement = load_model(MODEL_PATH)
model_commerce = load_model(COMMERCE_MODEL_PATH)

class ScrollFeature(BaseModel):
    position_index: int
    user_category_score: float
    video_quality: float
    has_products: int
    hour_of_day: int
    completion_rate: float
    # New Intelligence Features
    expected_ctr_at_position: float = 0.0
    session_velocity: float = 1.0
    session_dwell_time: float = 0.0

class BatchRequest(BaseModel):
    items: list[ScrollFeature]

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "models": {
            "engagement": "v1_shadow" if model_engagement else "missing",
            "commerce": "commerce_v1" if model_commerce else "missing"
        }
    }

@app.post("/predict")
def predict(batch: BatchRequest):
    """
    Multi-Task Inference: Predicts both Reel Engagement and Product Intent.
    """
    start_time = time.time()
    
    try:
        # 1. Convert batch to numpy array
        X = np.array([
            [
                item.position_index,
                item.user_category_score,
                item.video_quality,
                item.has_products,
                item.hour_of_day,
                item.completion_rate,
                item.expected_ctr_at_position,
                item.session_velocity,
                item.session_dwell_time
            ]
            for item in batch.items
        ])

        # 2. Convert to DataFrame to avoid "X does not have valid feature names" warnings
        X_df = ensure_feature_frame(pd.DataFrame(X, columns=FEATURE_COLUMNS))

        # 3. Parallel Prediction Logic
        results = {}
        
        if model_engagement:
            results["engagement_scores"] = model_engagement.predict_proba(X_df)[:, 1].tolist()
        
        if model_commerce:
            results["commerce_scores"] = model_commerce.predict_proba(X_df)[:, 1].tolist()

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
            fast_skip_score = max(0.0, min(1.0, 1.0 - ((0.7 * engagement) + (0.3 * completion_rate))))
            final_rank_score = max(0.0, min(1.0,
                (0.35 * qualified_view) +
                (0.20 * completion_score) +
                (0.15 * save_score) +
                (0.15 * product_click_score) +
                (0.10 * freshness_hint) -
                (0.20 * fast_skip_score)
            ))

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
            "ranker_version": "phase5-home-multitask-v1",
            "latency_ms": round(latency_ms, 2),
            "count": len(batch.items)
        }

    except Exception as e:
        print(f"❌ Multi-Task Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
