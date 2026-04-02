import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from feature_engineering import ensure_training_frame


load_dotenv()

OUTPUT_PATH = Path("data/processed/real_training_v1.parquet")


def build_training_dataset(limit=50000):
    """
    Build a real training dataset from production logs.

    Phase 1 goals:
    - Keep training/inference/evaluation on one shared feature contract.
    - Attribute interactions to impressions more reliably.
    - Emit both engagement and commerce labels from real logs.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not found.")
        return None

    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    engine = create_engine(db_url)

    print("🛰️  Fetching raw data from production DB...")

    query = f"""
    WITH base_impressions AS (
        SELECT
            imp.id AS impression_id,
            imp.user_id,
            imp.scroll_id,
            imp.session_id,
            imp.position_index,
            imp.feed_type,
            imp.created_at AS impressed_at,
            imp.ml_shadow_score,
            imp.ml_commerce_score,
            imp.heuristic_score,
            COALESCE(s.content_category, s.category) AS content_category,
            s.category,
            COALESCE(s.quality_score, 0) AS video_quality,
            CASE
                WHEN s.products IS NOT NULL AND cardinality(s.products) > 0 THEN 1
                ELSE 0
            END AS has_products,
            EXTRACT(HOUR FROM imp.created_at) AS hour_of_day
        FROM scroll_impressions imp
        JOIN scrolls s ON imp.scroll_id = s.id
        ORDER BY imp.created_at DESC
        LIMIT {int(limit)}
    )
    SELECT
        bi.impression_id,
        bi.user_id,
        bi.scroll_id,
        bi.position_index,
        bi.feed_type,
        bi.impressed_at,
        bi.ml_shadow_score,
        bi.ml_commerce_score,
        bi.heuristic_score,
        bi.category,
        bi.content_category,
        bi.video_quality,
        bi.has_products,
        bi.hour_of_day,
        COALESCE(uca.score, 0.0) AS user_category_score,
        COALESCE(pb.expected_ctr, 0.05) AS expected_ctr_at_position,
        COALESCE(
            CASE
                WHEN ss.impressions > 0 THEN ss.completions::float / ss.impressions::float
                ELSE 0
            END,
            0.0
        ) AS completion_rate,
        COALESCE(session_ctx.session_velocity, 1.0) AS session_velocity,
        COALESCE(session_ctx.session_dwell_time, 0.0) AS session_dwell_time,
        COALESCE(ir.is_qualified_view, 0) AS is_qualified_view,
        COALESCE(ir.is_complete, 0) AS is_complete,
        COALESCE(ir.is_skip, 0) AS is_skip,
        COALESCE(ir.is_rewatch, 0) AS is_rewatch,
        COALESCE(ir.is_share, 0) AS is_share,
        COALESCE(ir.is_product_click, 0) AS is_product_click,
        COALESCE(ir.is_add_to_cart, 0) AS is_add_to_cart,
        COALESCE(ir.is_purchase, 0) AS is_purchase,
        COALESCE(ir.is_save, 0) AS is_save,
        COALESCE(ir.is_pause, 0) AS is_pause,
        CASE
            WHEN COALESCE(ir.is_qualified_view, 0) = 1
              OR COALESCE(ir.is_complete, 0) = 1
              OR COALESCE(ir.is_product_click, 0) = 1
              OR COALESCE(ir.is_add_to_cart, 0) = 1
              OR COALESCE(ir.is_purchase, 0) = 1
              OR COALESCE(ir.is_save, 0) = 1
            THEN 1 ELSE 0
        END AS is_click
    FROM base_impressions bi
    LEFT JOIN user_category_affinity uca
        ON uca.user_id = bi.user_id
       AND uca.category_id = bi.content_category
    LEFT JOIN position_bias_baseline pb
        ON pb.position_index = bi.position_index
    LEFT JOIN scroll_stats ss
        ON ss.scroll_id = bi.scroll_id
    LEFT JOIN LATERAL (
        SELECT
            MAX(CASE WHEN si.event_type = 1 THEN 1 ELSE 0 END) AS is_qualified_view,
            MAX(CASE WHEN si.event_type = 2 THEN 1 ELSE 0 END) AS is_complete,
            MAX(CASE WHEN si.event_type = 3 THEN 1 ELSE 0 END) AS is_skip,
            MAX(CASE WHEN si.event_type = 4 THEN 1 ELSE 0 END) AS is_rewatch,
            MAX(CASE WHEN si.event_type = 5 THEN 1 ELSE 0 END) AS is_share,
            MAX(CASE WHEN si.event_type = 6 THEN 1 ELSE 0 END) AS is_product_click,
            MAX(CASE WHEN si.event_type = 7 THEN 1 ELSE 0 END) AS is_add_to_cart,
            MAX(CASE WHEN si.event_type = 8 THEN 1 ELSE 0 END) AS is_purchase,
            MAX(CASE WHEN si.event_type = 9 THEN 1 ELSE 0 END) AS is_save,
            MAX(CASE WHEN si.event_type = 13 THEN 1 ELSE 0 END) AS is_pause
        FROM scroll_interactions si
        WHERE
            (
                si.impression_id = bi.impression_id
                OR (
                    si.impression_id IS NULL
                    AND si.user_id = bi.user_id
                    AND si.scroll_id = bi.scroll_id
                    AND (bi.session_id IS NULL OR si.session_id = bi.session_id)
                    AND si.created_at >= bi.impressed_at
                    AND si.created_at < bi.impressed_at + interval '1 hour'
                )
            )
    ) ir ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            GREATEST(
                COUNT(*) FILTER (
                    WHERE prev_imp.created_at >= bi.impressed_at - interval '5 minutes'
                )::float / 5.0,
                0.0
            ) AS session_velocity,
            GREATEST(
                EXTRACT(EPOCH FROM (bi.impressed_at - MIN(prev_imp.created_at))),
                0
            ) AS session_dwell_time
        FROM scroll_impressions prev_imp
        WHERE prev_imp.user_id = bi.user_id
          AND prev_imp.created_at <= bi.impressed_at
          AND (
                (bi.session_id IS NOT NULL AND prev_imp.session_id = bi.session_id)
                OR (bi.session_id IS NULL AND prev_imp.created_at >= bi.impressed_at - interval '30 minutes')
          )
    ) session_ctx ON TRUE
    ORDER BY bi.impressed_at DESC;
    """

    try:
        df = pd.read_sql(query, engine)
    except SQLAlchemyError as exc:
        print("❌ Failed to build real dataset from Postgres.")
        print(f"   Reason: {exc}")
        return None

    if df.empty:
        print("⚠️  No data found in scroll_impressions yet. Start scrolling in the app!")
        return None

    for column in ["impression_id", "user_id", "scroll_id"]:
        if column in df.columns:
            df[column] = df[column].astype(str)

    if "impressed_at" in df.columns:
        df["impressed_at"] = pd.to_datetime(df["impressed_at"])

    df = ensure_training_frame(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"✅ Dataset built: {OUTPUT_PATH}")
    print(f"📦 Rows: {len(df)}")
    print(f"📈 Click Rate: {df['is_click'].mean():.2%}")
    print(f"🛒 Purchase Rate: {df['is_purchase'].mean():.2%}")
    print(f"👀 Qualified View Rate: {df['is_qualified_view'].mean():.2%}")

    return df


if __name__ == "__main__":
    build_training_dataset()
