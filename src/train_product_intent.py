import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from feature_engineering import ensure_feature_frame

def train_product_intent_model(dataset_path="data/processed/synthetic_training.parquet"):
    """
    Trains a specialized model to predict HIGH-INTENT commerce behavior.
    While v1_shadow predicts clicks, this model predicts Product Interest/Purchase.
    """
    print(f"📦 Loading dataset for Product Intent training from {dataset_path}...")
    df = pd.read_parquet(dataset_path)

    # 1. Feature Selection (Same list as V1 for inference simplicity, 
    # but the model will learn different weights)
    target = "is_purchase" # Predicting shoppability/high-intent

    X = ensure_feature_frame(df)
    y = df[target]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model Definition - LightGBM
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.03, # Slightly slower learning for rarer events
        "feature_fraction": 0.8,
        "is_unbalance": True, # Purchases are rarer than clicks, so we balance weights
        "force_col_wise": True,
        "verbosity": -1
    }

    print("🚀 Training Commerce Intelligence Model (Product Click Intent)...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"✅ Training complete.")
    print(f"   Commerce AUC-ROC: {auc:.4f}")
    
    # 5. Save Artifacts
    os.makedirs("models", exist_ok=True)
    model_path = "models/commerce_v1.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Product Intent Model saved to: {model_path}")

if __name__ == "__main__":
    train_product_intent_model()
