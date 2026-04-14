from pathlib import Path

import joblib
import lightgbm as lgb
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from .feature_engineering import ensure_feature_frame
except ImportError:
    from feature_engineering import ensure_feature_frame


REAL_DATASET_PATH = Path("data/processed/real_training_v1.parquet")
SYNTHETIC_DATASET_PATH = Path("data/processed/synthetic_training.parquet")
MIN_POSITIVE_EXAMPLES = 12


def _resolve_dataset_path(dataset_path=None):
    if dataset_path:
        return Path(dataset_path)
    return REAL_DATASET_PATH if REAL_DATASET_PATH.exists() else SYNTHETIC_DATASET_PATH


def _get_label_stats(series):
    normalized = series.fillna(0).astype(int)
    positive_count = int((normalized == 1).sum())
    negative_count = int((normalized == 0).sum())
    return normalized, positive_count, negative_count


def _load_dataset_for_target(target, dataset_path=None):
    primary_path = _resolve_dataset_path(dataset_path)
    candidate_paths = [primary_path]

    if primary_path != SYNTHETIC_DATASET_PATH:
        candidate_paths.append(SYNTHETIC_DATASET_PATH)

    for path in candidate_paths:
        if not path.exists():
            continue

        print(f"Loading dataset from {path}...")
        df = pd.read_parquet(path)
        y, positive_count, negative_count = _get_label_stats(df[target])

        print(
            f"   Target={target} positives={positive_count} negatives={negative_count} "
            f"positive_rate={(positive_count / max(len(df), 1)):.2%}"
        )

        if positive_count >= MIN_POSITIVE_EXAMPLES and negative_count >= MIN_POSITIVE_EXAMPLES:
            return df, y, str(path), str(path) != str(primary_path)

        print(
            f"⚠️ Insufficient labeled examples in {path} for {target}. "
            f"Need at least {MIN_POSITIVE_EXAMPLES} positives and negatives."
        )

    raise RuntimeError(
        f"Unable to find a usable dataset for target={target}. "
        "Generate more real feed traffic or keep the synthetic bootstrap dataset available."
    )


def train_v1_shadow(dataset_path=None):
    """
    Train the engagement model.

    Prefer real production data when it has enough positives; otherwise
    fall back to the synthetic bootstrap dataset.
    """
    target = "is_click"
    df, y, resolved_dataset_path, used_fallback_dataset = _load_dataset_for_target(target, dataset_path=dataset_path)

    X = ensure_feature_frame(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "force_col_wise": True,
        "verbosity": -1,
    }

    print("🚀 Training LightGBM model...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("✅ Training complete.")
    print(f"   Dataset: {resolved_dataset_path}")
    print(f"   Target: {target}")
    print(f"   AUC-ROC: {auc:.4f}")

    os.makedirs("models", exist_ok=True)
    model_path = "models/v1_shadow.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")

    return model


def train_v1_shadow_with_metadata(dataset_path=None, output_path="models/v1_shadow.joblib"):
    """
    Train engagement model and return promotion-safe metadata.
    """
    target = "is_click"
    df, y, resolved_dataset_path, used_fallback_dataset = _load_dataset_for_target(
        target,
        dataset_path=dataset_path,
    )

    X = ensure_feature_frame(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "force_col_wise": True,
        "verbosity": -1,
    }

    print("🚀 Training LightGBM model...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    metadata = {
        "task": "engagement",
        "dataset_path": resolved_dataset_path,
        "used_fallback_dataset": used_fallback_dataset,
        "target": target,
        "auc_roc": float(auc),
        "row_count": int(len(df)),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == 0).sum()),
        "model_path": str(output_path),
    }

    print("✅ Training complete.")
    print(f"   Dataset: {resolved_dataset_path}")
    print(f"   Target: {target}")
    print(f"   AUC-ROC: {auc:.4f}")
    print(f"✅ Model saved to: {output_path}")

    return model, metadata


if __name__ == "__main__":
    train_v1_shadow()
