import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def run_calibration_check(dataset_path="data/processed/real_training_v1.parquet", output_dir="data/evaluation"):
    """
    Evaluates if ML predicted probabilities match real-world frequencies.
    If probability=0.8, we should see 80% click rate.
    """
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(dataset_path)
    
    # Filter for Shadow Mode data
    shadow_df = df[df['ml_shadow_score'].notnull()].copy()
    if shadow_df.empty:
        print("⚠️ No data for calibration check.")
        return

    if shadow_df['is_click'].nunique() < 2:
        print("⚠️ Need both positive and negative click labels for calibration.")
        return

    shadow_df['ml_shadow_score'] = shadow_df['ml_shadow_score'].astype(float)
    y_true = shadow_df['is_click']
    y_prob = shadow_df['ml_shadow_score']

    # 1. Brier Score (Lower is better, 0.0 is perfect)
    brier = brier_score_loss(y_true, y_prob)
    
    # 2. Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    # 3. Expected Calibration Error (ECE) - Weighted average of (prob_pred - prob_true)
    ece = np.mean(np.abs(prob_pred - prob_true))

    print("\n--- MODEL CALIBRATION REPORT ---")
    print(f"Brier Score (MSE)          : {brier:.4f} (Lower = More Accurate)")
    print(f"Expected Calibration Error : {ece:.4f} (Lower = More Reliable)")
    
    # Visual Plot
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label=f"ML Model (Brier={brier:.3f})")

    plt.ylabel("Actual Click Frequency")
    plt.xlabel("Predicted Click Probability")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend(loc="lower right")
    
    plot_path = os.path.join(output_dir, "calibration_curve.png")
    plt.savefig(plot_path)
    print(f"📉 Calibration plot saved to {plot_path}")

    # Final Verdict
    if ece < 0.05:
        print("✅ Verdict: HIGHLY CALIBRATED. Model probabilities can be trusted.")
    elif ece < 0.15:
        print("⚠️ Verdict: MODERATELY CALIBRATED. Model is directionally correct but tends to over/under-estimate.")
    else:
        print("🚨 Verdict: MISCALIBRATED. Retraining with temperature scaling/isotonic regression recommended.")

if __name__ == "__main__":
    run_calibration_check()
