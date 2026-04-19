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


def _to_binary(series_or_scalar, row_count):
    if isinstance(series_or_scalar, pd.Series):
        return series_or_scalar.fillna(0).astype(int)

    return pd.Series([series_or_scalar] * row_count).fillna(0).astype(int)


def _build_target_candidates(df):
    row_count = len(df)
    purchase = _to_binary(df.get("is_purchase", 0), row_count)
    checkout_start = _to_binary(df.get("is_checkout_start", 0), row_count)
    add_to_cart = _to_binary(df.get("is_add_to_cart", 0), row_count)
    product_click = _to_binary(df.get("is_product_click", 0), row_count)
    product_view = _to_binary(df.get("is_product_view", 0), row_count)

    commerce_intent = (
        (purchase == 1)
        | (checkout_start == 1)
        | (add_to_cart == 1)
        | (product_click == 1)
        | (product_view == 1)
    ).astype(int)

    return [
        ("is_purchase", purchase),
        ("commerce_intent", commerce_intent),
        ("is_checkout_start", checkout_start),
        ("is_add_to_cart", add_to_cart),
        ("is_product_click", product_click),
        ("is_product_view", product_view),
    ]


def _select_target(df):
    for target_name, target_values in _build_target_candidates(df):
        positive_count = int((target_values == 1).sum())
        negative_count = int((target_values == 0).sum())

        print(
            f"   Candidate target={target_name} positives={positive_count} negatives={negative_count} "
            f"positive_rate={(positive_count / max(len(df), 1)):.2%}"
        )

        if positive_count >= MIN_POSITIVE_EXAMPLES and negative_count >= MIN_POSITIVE_EXAMPLES:
            return target_name, target_values

    return None, None


def _load_dataset_and_target(dataset_path=None):
    primary_path = _resolve_dataset_path(dataset_path)
    candidate_paths = [primary_path]

    if primary_path != SYNTHETIC_DATASET_PATH:
        candidate_paths.append(SYNTHETIC_DATASET_PATH)

    for path in candidate_paths:
        if not path.exists():
            continue

        print(f"📦 Loading dataset for Product Intent training from {path}...")
        df = pd.read_parquet(path)
        target_name, y = _select_target(df)

        if target_name is not None:
            return df, y, target_name, str(path), str(path) != str(primary_path)

        print(
            f"⚠️ No usable commerce target found in {path}. "
            f"Need at least {MIN_POSITIVE_EXAMPLES} positives."
        )

    raise RuntimeError(
        "Unable to train commerce model from available datasets. "
        "Generate more product clicks/add-to-cart/purchase events or keep the synthetic dataset available."
    )


def train_product_intent_model(dataset_path=None):
    """
    Train the commerce model.

    Prefer pure purchases when enough real positives exist. Otherwise use the
    strongest available commerce proxy from real data, and only fall back to
    synthetic bootstrap data when the real dataset is too sparse.
    """
    df, y, target_name, resolved_dataset_path, used_fallback_dataset = _load_dataset_and_target(dataset_path=dataset_path)
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
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "is_unbalance": True,
        "force_col_wise": True,
        "verbosity": -1,
    }

    print("🚀 Training Commerce Intelligence Model...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("✅ Training complete.")
    print(f"   Dataset: {resolved_dataset_path}")
    print(f"   Target: {target_name}")
    print(f"   Commerce AUC-ROC: {auc:.4f}")

    os.makedirs("models", exist_ok=True)
    model_path = "models/commerce_v1.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Product Intent Model saved to: {model_path}")

    return model


def train_product_intent_with_metadata(dataset_path=None, output_path="models/commerce_v1.joblib"):
    """
    Train commerce model and return promotion-safe metadata.
    """
    df, y, target_name, resolved_dataset_path, used_fallback_dataset = _load_dataset_and_target(
        dataset_path=dataset_path
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
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "is_unbalance": True,
        "force_col_wise": True,
        "verbosity": -1,
    }

    print("🚀 Training Commerce Intelligence Model...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    metadata = {
        "task": "commerce",
        "dataset_path": resolved_dataset_path,
        "used_fallback_dataset": used_fallback_dataset,
        "target": target_name,
        "auc_roc": float(auc),
        "row_count": int(len(df)),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == 0).sum()),
        "model_path": str(output_path),
    }

    print("✅ Training complete.")
    print(f"   Dataset: {resolved_dataset_path}")
    print(f"   Target: {target_name}")
    print(f"   Commerce AUC-ROC: {auc:.4f}")
    print(f"✅ Product Intent Model saved to: {output_path}")

    return model, metadata


if __name__ == "__main__":
    train_product_intent_model()
