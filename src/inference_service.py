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

app = FastAPI(
    title="Scroll ML Inference Service",
    description="Batch scoring for Shadow Mode ranking.",
    version="1.0.0",
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

# Feature list must match the training script exactly
# Updated for "Controlled Intelligence" Phase 2
FEATURES = [
    "position_index", 
    "user_category_score", 
    "video_quality", 
    "has_products", 
    "hour_of_day", 
    "completion_rate",
    "expected_ctr_at_position",
    "session_velocity",
    "session_dwell_time"
]

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
        X_df = pd.DataFrame(X, columns=FEATURES)

        # 3. Parallel Prediction Logic
        results = {}
        
        if model_engagement:
            results["engagement_scores"] = model_engagement.predict_proba(X_df)[:, 1].tolist()
        
        if model_commerce:
            results["commerce_scores"] = model_commerce.predict_proba(X_df)[:, 1].tolist()

        latency_ms = (time.time() - start_time) * 1000

        return {
            **results,
            "latency_ms": round(latency_ms, 2),
            "count": len(batch.items)
        }

    except Exception as e:
        print(f"❌ Multi-Task Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
