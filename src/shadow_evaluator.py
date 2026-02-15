import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score

def run_evaluation(dataset_path="data/processed/real_training_v1.parquet"):
    """
    Evaluates ML Shadow Mode performance vs Manual Heuristic logic.
    This uses the audit logs collected in the production DB.
    """
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}. Run dataset_builder.py first.")
        return

    df = pd.read_parquet(dataset_path)
    
    # Filter for rows that have both scores (Shadow Mode data)
    shadow_df = df[df['ml_shadow_score'].notnull() & df['heuristic_score'].notnull()].copy()
    
    if shadow_df.empty:
        print("⚠️ No Shadow Mode logs found yet. Please browse the app to collect data.")
        return

    # Convert to numeric
    shadow_df['ml_shadow_score'] = shadow_df['ml_shadow_score'].astype(float)
    shadow_df['ml_commerce_score'] = shadow_df['ml_commerce_score'].astype(float)
    shadow_df['heuristic_score'] = shadow_df['heuristic_score'].astype(float)
    
    print(f"📋 Evaluating {len(shadow_df)} Shadow Mode impressions...")
    print(f"📊 Label Distribution (Clicks): {shadow_df['is_click'].value_counts().to_dict()}")
    
    if shadow_df['is_click'].nunique() < 2:
        print("⚠️ Warning: Only one class found in shadow data. AUC-ROC cannot be calculated.")
        print("💡 Strategy: Interaction more with the app to generate a mix of clicks/skips.")
        return

    # 1. AUC-ROC (Ability to distinguish Clicks vs Non-clicks)
    ml_auc = roc_auc_score(shadow_df['is_click'], shadow_df['ml_shadow_score'])
    com_auc = roc_auc_score(shadow_df['is_click'], shadow_df['ml_commerce_score'])
    heu_auc = roc_auc_score(shadow_df['is_click'], shadow_df['heuristic_score'])
    
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Metric       | ML Engagement | ML Commerce | Heuristic")
    print(f"-------------|---------------|-------------|----------")
    print(f"AUC-ROC      | {ml_auc:.4f}        | {com_auc:.4f}      | {heu_auc:.4f}")
    
    # 2. Lift Analysis
    lift = (ml_auc - heu_auc) / heu_auc if heu_auc > 0 else 0
    print(f"\n🚀 Current Engagement Lift: {lift:+.2%}")
    
    if ml_auc > heu_auc:
        print("💡 Recommendation: ML is outperforming. Consider increasing shadow weight.")
    else:
        print("🛠️ Recommendation: Heuristic is stronger. Model requires more training on real features.")

if __name__ == "__main__":
    run_evaluation()
