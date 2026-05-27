from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

try:
    from .db_config import resolve_sqlalchemy_database_url
    from .feature_engineering import ensure_training_frame
    from .recommendation_events import EVENT_TYPES
except ImportError:
    from db_config import resolve_sqlalchemy_database_url
    from feature_engineering import ensure_training_frame
    from recommendation_events import EVENT_TYPES


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
    db_url = resolve_sqlalchemy_database_url()
    if not db_url:
        print("ERROR: DATABASE_URL not found.")
        return None

    engine = create_engine(db_url)

    print("Fetching raw data from production DB...")

    query = f"""
    WITH base_impressions AS (
        SELECT
            imp.id AS impression_id,
            imp.user_id,
            imp.scroll_id,
            imp.product_id,
            COALESCE(imp.item_type, CASE WHEN imp.product_id IS NOT NULL THEN 2 ELSE 1 END) AS item_type,
            imp.session_id,
            imp.position_index,
            imp.feed_type,
            imp.created_at AS impressed_at,
            imp.ml_shadow_score,
            imp.ml_commerce_score,
            imp.heuristic_score,
            imp.recommendation_run_id,
            imp.ranking_version,
            imp.candidate_source,
            imp.recommendation_profile_kind,
            COALESCE(imp.retrieval_score::float, 0.0) AS retrieval_score,
            COALESCE(imp.semantic_score::float, 0.0) AS semantic_score,
            COALESCE(s.content_category, p.content_category, s.category, p.category) AS content_category,
            COALESCE(s.category, p.category) AS category,
            COALESCE(s.user_id, p.user_id) AS creator_id,
            CASE
                WHEN COALESCE(imp.item_type, CASE WHEN imp.product_id IS NOT NULL THEN 2 ELSE 1 END) = 2
                    THEN COALESCE(p.average_rating::float, 0.0)
                ELSE COALESCE(s.quality_score::float, 0.0)
            END AS video_quality,
            CASE
                WHEN COALESCE(imp.item_type, CASE WHEN imp.product_id IS NOT NULL THEN 2 ELSE 1 END) = 2 THEN 1
                WHEN s.products IS NOT NULL AND cardinality(s.products) > 0 THEN 1
                ELSE 0
            END AS has_products,
            EXTRACT(HOUR FROM imp.created_at) AS hour_of_day,
            CASE
                WHEN COALESCE(imp.item_type, CASE WHEN imp.product_id IS NOT NULL THEN 2 ELSE 1 END) = 2
                    THEN LEAST(
                        1.0,
                        (
                            COALESCE(igs.clicks, 0)::float
                            + COALESCE(igs.product_views, 0)::float
                            + (COALESCE(igs.carts, 0)::float * 3.0)
                            + (COALESCE(igs.purchases, 0)::float * 6.0)
                        ) / GREATEST(COALESCE(igs.impressions, 0)::float, 1.0)
                    )
                ELSE COALESCE(
                    CASE
                        WHEN ss.impressions > 0 THEN ss.completions::float / ss.impressions::float
                        ELSE 0
                    END,
                    0.0
                )
            END AS completion_rate,
            CASE
                WHEN COALESCE(imp.item_type, CASE WHEN imp.product_id IS NOT NULL THEN 2 ELSE 1 END) = 2
                    THEN COALESCE(
                        (COALESCE(igs.clicks, 0)::float + COALESCE(igs.product_views, 0)::float)
                        / NULLIF(COALESCE(igs.impressions, 0)::float, 0.0),
                        0.0
                    )
                ELSE COALESCE(
                    COALESCE(ss.qualified_views, 0)::float / NULLIF(COALESCE(ss.impressions, 0)::float, 0.0),
                    0.0
                )
            END AS global_ctr,
            CASE
                WHEN COALESCE(imp.item_type, CASE WHEN imp.product_id IS NOT NULL THEN 2 ELSE 1 END) = 2
                    THEN COALESCE(
                        COALESCE(igs.purchases, 0)::float / NULLIF(COALESCE(igs.impressions, 0)::float, 0.0),
                        0.0
                    )
                ELSE COALESCE(
                    COALESCE(ss.purchases, 0)::float / NULLIF(COALESCE(ss.impressions, 0)::float, 0.0),
                    0.0
                )
            END AS global_conversion_rate,
            CASE
                WHEN COALESCE(imp.item_type, CASE WHEN imp.product_id IS NOT NULL THEN 2 ELSE 1 END) = 2
                    THEN LEAST(
                        1.0,
                        LN(
                            1
                            + GREATEST(COALESCE(p.total_sales_count, 0), 0)
                            + GREATEST(COALESCE(p.review_count, 0), 0)
                            + GREATEST(COALESCE(p.likes, 0), 0)
                            + GREATEST(COALESCE(p.save_count, 0), 0)
                        ) / 5.0
                    )
                ELSE LEAST(
                    1.0,
                    LN(
                        1
                        + GREATEST(COALESCE(s.views, 0), 0)
                        + GREATEST(COALESCE(s.likes, 0), 0)
                        + GREATEST(COALESCE(s.save_count, 0), 0)
                    ) / 5.0
                )
            END AS social_proof_score,
            CASE
                WHEN COALESCE(imp.item_type, CASE WHEN imp.product_id IS NOT NULL THEN 2 ELSE 1 END) = 2
                    THEN COALESCE(p.created_at, imp.created_at)
                ELSE COALESCE(s.created_at, imp.created_at)
            END AS content_created_at
        FROM scroll_impressions imp
        LEFT JOIN scrolls s
            ON imp.scroll_id = s.id
        LEFT JOIN products p
            ON imp.product_id = p.id
        LEFT JOIN scroll_stats ss
            ON ss.scroll_id = imp.scroll_id
        LEFT JOIN item_global_stats igs
            ON igs.item_id = COALESCE(imp.product_id, imp.scroll_id)
           AND igs.item_type = COALESCE(imp.item_type, CASE WHEN imp.product_id IS NOT NULL THEN 2 ELSE 1 END)
        WHERE imp.scroll_id IS NOT NULL OR imp.product_id IS NOT NULL
        ORDER BY imp.created_at DESC
        LIMIT {int(limit)}
    )
    SELECT
        bi.impression_id,
        bi.user_id,
        bi.scroll_id,
        bi.product_id,
        bi.item_type,
        bi.position_index,
        bi.feed_type,
        bi.impressed_at,
        bi.ml_shadow_score,
        bi.ml_commerce_score,
        bi.heuristic_score,
        bi.recommendation_run_id,
        bi.ranking_version,
        bi.candidate_source,
        bi.recommendation_profile_kind,
        bi.category,
        bi.content_category,
        bi.video_quality,
        bi.has_products,
        bi.hour_of_day,
        COALESCE(uca.score, 0.0) AS user_category_score,
        COALESCE(ucr.score, 0.0) AS creator_affinity_score,
        COALESCE(pb.expected_ctr, 0.05) AS expected_ctr_at_position,
        COALESCE(bi.completion_rate, 0.0) AS completion_rate,
        COALESCE(session_ctx.session_velocity, 1.0) AS session_velocity,
        COALESCE(session_ctx.session_dwell_time, 0.0) AS session_dwell_time,
        CASE WHEN bi.item_type = 2 THEN 1 ELSE 0 END AS is_product,
        COALESCE(
            1.0 / (
                1.0
                + (
                    EXTRACT(EPOCH FROM (bi.impressed_at - bi.content_created_at)) / 3600.0
                ) / CASE WHEN bi.item_type = 2 THEN 72.0 ELSE 24.0 END
            ),
            0.0
        ) AS content_freshness,
        COALESCE(bi.global_ctr, 0.0) AS global_ctr,
        COALESCE(bi.global_conversion_rate, 0.0) AS global_conversion_rate,
        COALESCE(bi.social_proof_score, 0.0) AS social_proof_score,
        COALESCE(bi.semantic_score, 0.0) AS semantic_score,
        COALESCE(bi.retrieval_score, 0.0) AS retrieval_score,
        CASE
            WHEN bi.semantic_score > 0 OR bi.retrieval_score > 0 THEN 1
            ELSE 0
        END AS has_semantic_candidate,
        CASE
            WHEN bi.recommendation_profile_kind = 'behavior_14d' THEN 1.0
            WHEN bi.recommendation_profile_kind = 'behavior_7d' THEN 0.9
            WHEN bi.recommendation_profile_kind = 'session' THEN 0.75
            WHEN bi.recommendation_profile_kind = 'declared_interest' THEN 0.55
            WHEN bi.recommendation_profile_kind = 'manual' THEN 0.4
            ELSE 0.0
        END AS semantic_profile_strength,
        COALESCE(ir.is_qualified_view, 0) AS is_qualified_view,
        COALESCE(ir.is_complete, 0) AS is_complete,
        COALESCE(ir.is_skip, 0) AS is_skip,
        COALESCE(ir.is_rewatch, 0) AS is_rewatch,
        COALESCE(ir.is_product_view, 0) AS is_product_view,
        COALESCE(ir.is_share, 0) AS is_share,
        COALESCE(ir.is_product_click, 0) AS is_product_click,
        COALESCE(ir.is_add_to_cart, 0) AS is_add_to_cart,
        COALESCE(ir.is_checkout_start, 0) AS is_checkout_start,
        COALESCE(ir.is_purchase, 0) AS is_purchase,
        COALESCE(ir.is_save, 0) AS is_save,
        COALESCE(ir.is_pause, 0) AS is_pause,
        CASE
            WHEN COALESCE(ir.is_qualified_view, 0) = 1
              OR COALESCE(ir.is_complete, 0) = 1
              OR COALESCE(ir.is_product_view, 0) = 1
              OR COALESCE(ir.is_product_click, 0) = 1
              OR COALESCE(ir.is_add_to_cart, 0) = 1
              OR COALESCE(ir.is_checkout_start, 0) = 1
              OR COALESCE(ir.is_purchase, 0) = 1
              OR COALESCE(ir.is_save, 0) = 1
            THEN 1 ELSE 0
        END AS is_click
    FROM base_impressions bi
    LEFT JOIN user_category_affinity uca
        ON uca.user_id = bi.user_id
       AND uca.category_id = bi.content_category
    LEFT JOIN user_creator_affinity ucr
        ON ucr.user_id = bi.user_id
       AND ucr.creator_id = bi.creator_id
    LEFT JOIN position_bias_baseline pb
        ON pb.position_index = bi.position_index
    LEFT JOIN LATERAL (
        SELECT
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["QUALIFIED_VIEW"]} THEN 1 ELSE 0 END) AS is_qualified_view,
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["COMPLETE"]} THEN 1 ELSE 0 END) AS is_complete,
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["SKIP"]} THEN 1 ELSE 0 END) AS is_skip,
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["REWATCH"]} THEN 1 ELSE 0 END) AS is_rewatch,
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["PRODUCT_VIEW"]} THEN 1 ELSE 0 END) AS is_product_view,
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["SHARE"]} THEN 1 ELSE 0 END) AS is_share,
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["PRODUCT_CLICK"]} THEN 1 ELSE 0 END) AS is_product_click,
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["ADD_TO_CART"]} THEN 1 ELSE 0 END) AS is_add_to_cart,
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["CHECKOUT_START"]} THEN 1 ELSE 0 END) AS is_checkout_start,
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["PURCHASE"]} THEN 1 ELSE 0 END) AS is_purchase,
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["SAVE"]} THEN 1 ELSE 0 END) AS is_save,
            MAX(CASE WHEN si.event_type = {EVENT_TYPES["SCROLL_PAUSE"]} THEN 1 ELSE 0 END) AS is_pause
        FROM scroll_interactions si
        WHERE
            (
                si.impression_id = bi.impression_id
                OR (
                    si.impression_id IS NULL
                    AND si.user_id = bi.user_id
                    AND (
                        (bi.item_type = 1 AND si.scroll_id = bi.scroll_id)
                        OR (bi.item_type = 2 AND si.product_id = bi.product_id)
                    )
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
        print("ERROR: Failed to build real dataset from Postgres.")
        print(f"   Reason: {exc}")
        return None

    if df.empty:
        print("WARNING: No data found in scroll_impressions yet. Start scrolling in the app.")
        return None

    for column in [
        "impression_id",
        "user_id",
        "scroll_id",
        "product_id",
        "recommendation_run_id",
        "candidate_source",
        "recommendation_profile_kind",
        "ranking_version",
    ]:
        if column in df.columns:
            df[column] = df[column].astype(str)

    if "impressed_at" in df.columns:
        df["impressed_at"] = pd.to_datetime(df["impressed_at"])

    df = ensure_training_frame(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Dataset built: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Click Rate: {df['is_click'].mean():.2%}")
    print(f"Purchase Rate: {df['is_purchase'].mean():.2%}")
    print(f"Qualified View Rate: {df['is_qualified_view'].mean():.2%}")

    return df


if __name__ == "__main__":
    build_training_dataset()
