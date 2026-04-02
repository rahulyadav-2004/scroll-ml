import os

from train_hybrid_model import train_v1_shadow


def train_click_model():
    """
    Phase 1 production entrypoint for engagement training.

    Prefer real training data when present; fall back to synthetic bootstrap data
    so local development still works.
    """
    real_dataset = "data/processed/real_training_v1.parquet"
    synthetic_dataset = "data/processed/synthetic_training.parquet"

    dataset_path = real_dataset if os.path.exists(real_dataset) else synthetic_dataset
    print(f"🎯 Training click model from: {dataset_path}")
    return train_v1_shadow(dataset_path=dataset_path)


if __name__ == "__main__":
    train_click_model()
