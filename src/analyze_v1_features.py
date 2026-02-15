import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_shadow_v1():
    """
    Observability Tool: Analyzes the v1_shadow model to understand feature importance.
    This helps the team understand why the model likes certain scrolls.
    """
    model_path = "models/v1_shadow.joblib"
    if not os.path.exists(model_path):
        print(f"❌ Model {model_path} not found.")
        return

    print(f"🧐 Analyzing Model: {model_path}")
    model = joblib.load(model_path)

    # 1. Extract Feature Importance
    # LightGBM records importance in its booster
    importance = model.feature_importances_
    
    # Feature names must match what was passed during trial (from train_hybrid_model.py)
    feature_names = [
        "position_index", 
        "user_category_score", 
        "video_quality", 
        "has_products", 
        "hour_of_day", 
        "completion_rate"
    ]

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    print("\n--- Feature Importance Table ---")
    print(importance_df)

    # 2. Basic Model Metadata
    print("\n--- Model Configuration ---")
    print(f"Algorithm: LightGBM Classifier")
    print(f"Parameters: {model.get_params()}")

    # 3. Create Visualization
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Shadow Model v1: Feature Importance (Observability)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    
    report_path = "reports/v1_importance.png"
    plt.savefig(report_path)
    print(f"\n✅ Observability report saved to: {report_path}")

    # 4. Calibration Check (Mock for now, but infrastructure is ready)
    # This is where we would check if predicted 0.2 means 20% real CTR
    print("\n💡 Recommendation for next phase:")
    print("If 'position_index' is too high, the model has learned 'Lazy Bias'. We should add position-bias correction.")

if __name__ == "__main__":
    analyze_shadow_v1()
