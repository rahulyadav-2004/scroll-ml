import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score

from feature_engineering import available_score_columns

def run_evaluation(dataset_path="data/processed/real_training_v1.parquet"):
    """
    Evaluates ML Shadow Mode performance vs Manual Heuristic logic.
    This uses the audit logs collected in the production DB.
    """
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}. Run dataset_builder.py first.")
        return

    df = pd.read_parquet(dataset_path)
    
    score_columns = available_score_columns(df)
    required_baseline_columns = {"ml_shadow_score", "heuristic_score"}

    if not required_baseline_columns.issubset(score_columns):
        print("⚠️ Required score columns are missing. Need at least ml_shadow_score and heuristic_score.")
        print(f"Available score columns: {score_columns}")
        return

    # Filter for rows that have both scores (Shadow Mode data)
    shadow_df = df[df['ml_shadow_score'].notnull() & df['heuristic_score'].notnull()].copy()
    
    if shadow_df.empty:
        print("⚠️ No Shadow Mode logs found yet. Please browse the app to collect data.")
        return

    # Convert to numeric
    shadow_df['ml_shadow_score'] = shadow_df['ml_shadow_score'].astype(float)
    shadow_df['heuristic_score'] = shadow_df['heuristic_score'].astype(float)
    if 'ml_commerce_score' in shadow_df.columns:
        shadow_df['ml_commerce_score'] = shadow_df['ml_commerce_score'].astype(float)
    
    print(f"📋 Evaluating {len(shadow_df)} Shadow Mode impressions...")
    print(f"📊 Label Distribution (Clicks): {shadow_df['is_click'].value_counts().to_dict()}")
    
    if shadow_df['is_click'].nunique() < 2:
        print("⚠️ Warning: Only one class found in shadow data. AUC-ROC cannot be calculated.")
        print("💡 Strategy: Interaction more with the app to generate a mix of clicks/skips.")
        return

    # 1. AUC-ROC (Ability to distinguish Clicks vs Non-clicks)
    ml_auc = roc_auc_score(shadow_df['is_click'], shadow_df['ml_shadow_score'])
    heu_auc = roc_auc_score(shadow_df['is_click'], shadow_df['heuristic_score'])
    com_auc = None
    if 'ml_commerce_score' in shadow_df.columns and shadow_df['ml_commerce_score'].notnull().any():
        com_auc = roc_auc_score(shadow_df['is_click'], shadow_df['ml_commerce_score'])
    
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Metric       | ML Engagement | ML Commerce | Heuristic")
    print(f"-------------|---------------|-------------|----------")
    commerce_display = f"{com_auc:.4f}" if com_auc is not None else "n/a"
    print(f"AUC-ROC      | {ml_auc:.4f}        | {commerce_display}      | {heu_auc:.4f}")
    
    # 2. Lift Analysis
    lift = (ml_auc - heu_auc) / heu_auc if heu_auc > 0 else 0
    print(f"\n🚀 Current Engagement Lift: {lift:+.2%}")
    
    if ml_auc > heu_auc:
        print("💡 Recommendation: ML is outperforming. Consider increasing shadow weight.")
    else:
        print("🛠️ Recommendation: Heuristic is stronger. Model requires more training on real features.")

if __name__ == "__main__":
    run_evaluation()
