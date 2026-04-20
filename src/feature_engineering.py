from __future__ import annotations

from typing import Iterable

import pandas as pd


CORE_FEATURE_COLUMNS = [
    "position_index",
    "user_category_score",
    "video_quality",
    "has_products",
    "hour_of_day",
    "completion_rate",
    "expected_ctr_at_position",
    "session_velocity",
    "session_dwell_time",
]

FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + [
    "is_product",
    "creator_affinity_score",
    "content_freshness",
    "global_ctr",
    "global_conversion_rate",
    "social_proof_score",
    "semantic_score",
    "retrieval_score",
    "has_semantic_candidate",
    "semantic_profile_strength",
]

LABEL_COLUMNS = [
    "is_click",
    "is_qualified_view",
    "is_complete",
    "is_skip",
    "is_rewatch",
    "is_product_view",
    "is_share",
    "is_product_click",
    "is_add_to_cart",
    "is_checkout_start",
    "is_purchase",
    "is_save",
    "is_pause",
]

SCORE_COLUMNS = [
    "ml_shadow_score",
    "ml_commerce_score",
    "heuristic_score",
]

METADATA_COLUMNS = [
    "impression_id",
    "user_id",
    "scroll_id",
    "product_id",
    "item_type",
    "feed_type",
    "category",
    "content_category",
    "impressed_at",
    "candidate_source",
    "recommendation_profile_kind",
    "recommendation_run_id",
    "ranking_version",
]

DATASET_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + SCORE_COLUMNS + LABEL_COLUMNS

DEFAULT_NUMERIC_VALUES = {
    "position_index": 0,
    "user_category_score": 0.0,
    "video_quality": 0.0,
    "has_products": 0,
    "hour_of_day": 0,
    "completion_rate": 0.0,
    "expected_ctr_at_position": 0.05,
    "session_velocity": 1.0,
    "session_dwell_time": 0.0,
    "is_product": 0,
    "creator_affinity_score": 0.0,
    "content_freshness": 0.0,
    "global_ctr": 0.0,
    "global_conversion_rate": 0.0,
    "social_proof_score": 0.0,
    "semantic_score": 0.0,
    "retrieval_score": 0.0,
    "has_semantic_candidate": 0,
    "semantic_profile_strength": 0.0,
    "ml_shadow_score": 0.0,
    "ml_commerce_score": 0.0,
    "heuristic_score": 0.0,
    "is_click": 0,
    "is_qualified_view": 0,
    "is_complete": 0,
    "is_skip": 0,
    "is_rewatch": 0,
    "is_product_view": 0,
    "is_share": 0,
    "is_product_click": 0,
    "is_add_to_cart": 0,
    "is_checkout_start": 0,
    "is_purchase": 0,
    "is_save": 0,
    "is_pause": 0,
}


def ensure_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature-only frame in the exact inference/training order."""
    return _ensure_columns(df, FEATURE_COLUMNS)


def ensure_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with all known dataset columns in stable order."""
    return _ensure_columns(df, DATASET_COLUMNS)


def available_score_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in SCORE_COLUMNS if col in df.columns]


def available_label_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in LABEL_COLUMNS if col in df.columns]


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()

    for column in columns:
        if column not in out.columns:
            out[column] = DEFAULT_NUMERIC_VALUES.get(column)

    for column, default_value in DEFAULT_NUMERIC_VALUES.items():
        if column not in out.columns:
            continue
        out[column] = out[column].fillna(default_value)

    return out[list(columns)]
