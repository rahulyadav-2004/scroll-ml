import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def train_v1_shadow(dataset_path="data/processed/synthetic_training.parquet"):
    """
    Trains the first 'Shadow Mode' model using synthetic data.
    This validates the machine logic, not the intelligence (yet).
    """
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)

    # 1. Feature Selection
    # We choose features that we are already collecting in the production DB
    features = [
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
    target = "is_click" # Using Click as the primary optimization target for v1

    X = df[features]
    y = df[target]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model Definition - LightGBM
    # Using small parameters for faster validation
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "force_col_wise": True,
        "verbosity": -1
    }

    print("🚀 Training LightGBM model...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"✅ Training complete. Performance (on synthetic data):")
    print(f"   AUC-ROC: {auc:.4f}")
    
    # 5. Save Artifacts
    os.makedirs("models", exist_ok=True)
    model_path = "models/v1_shadow.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")

if __name__ == "__main__":
    train_v1_shadow()
