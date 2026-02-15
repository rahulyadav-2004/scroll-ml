import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

def build_training_dataset():
    """
    Pulls real behavioral logs from Postgres and builds a dataset for ML training.
    Joints Impressions (exposure) with Interactions (outcome).
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not found.")
        return

    # 1. Connect to Database
    # Replacing postgresql:// with postgresql+psycopg2:// for SQLAlchemy compatibility
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    
    engine = create_engine(db_url)

    print("🛰️  Fetching raw data from production DB...")
    
    # 2. Main Query: Join Impressions and Interactions
    # We want to know: For every time a scroll was shown (Impression), did a Click happen?
    query = """
    SELECT 
        imp.id as impression_id,
        imp.user_id,
        imp.scroll_id,
        imp.position_index,
        imp.feed_type,
        imp.created_at as impressed_at,
        imp.ml_shadow_score,
        imp.ml_commerce_score,
        imp.heuristic_score,
        s.category,
        s.quality_score as video_quality,
        CASE WHEN s.products IS NOT NULL AND cardinality(s.products) > 0 THEN 1 ELSE 0 END as has_products,
        
        -- Outcome: 1 if there exists a 'Qualified Watch' (1) or 'Complete' (2) or 'Product Click' (6)
        CASE WHEN EXISTS (
            SELECT 1 FROM scroll_interactions int 
            WHERE int.scroll_id = imp.scroll_id 
            AND int.user_id = imp.user_id
            AND int.event_type IN (1, 2, 6)
            AND int.created_at >= imp.created_at
            AND int.created_at < imp.created_at + interval '1 hour'
        ) THEN 1 ELSE 0 END as is_click
        
    FROM scroll_impressions imp
    JOIN scrolls s ON imp.scroll_id = s.id
    ORDER BY imp.created_at DESC
    LIMIT 50000;
    """

    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("⚠️  No data found in scroll_impressions yet. Start scrolling in the app!")
        return

    dataset_path = "data/processed/real_training_v1.parquet"
    
    # 3. Data Cleaning for Parquet (UUIDs must be strings)
    for col in ['impression_id', 'user_id', 'scroll_id']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # 4. Add Contextual Features
    # Convert timestamp to hour of day
    df['impressed_at'] = pd.to_datetime(df['impressed_at'])
    df['hour_of_day'] = df['impressed_at'].dt.hour
    
    # 4. Save for Training/Analysis
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/real_training_v1.parquet"
    df.to_parquet(out_path, index=False)
    
    print(f"✅ Dataset built: {out_path}")
    print(f"📈 Real Click Rate in data: {df['is_click'].mean():.2%}")
    
    return df

if __name__ == "__main__":
    build_training_dataset()
