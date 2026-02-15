import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_synthetic_data(n_samples=10000):
    """
    Generates realistic synthetic data for validating the Scroll ML pipeline.
    Simulates position bias, category preference, and product interest.
    """
    np.random.seed(42)
    
    # 1. Base Context
    # position_index: 0 is top. Users click more at the top (Position Bias).
    position_index = np.random.randint(0, 15, n_samples)
    
    # user_category_score: 0 to 5 (Interest in the category of the scroll)
    user_category_score = np.random.uniform(0, 5, n_samples)
    
    # hour_of_day: 0 to 23 (Time of interaction)
    hour_of_day = np.random.randint(0, 24, n_samples)

    # 2. Content Features
    # video_quality: 0 to 1 (Resolution, lighting, etc.)
    video_quality = np.random.uniform(0, 1, n_samples)
    
    # has_products: boolean (Does it have shoppable items?)
    has_products = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    
    # 3. Behavioral Features (The Numerators)
    # completion_rate: 0 to 1
    completion_rate = np.random.uniform(0, 1, n_samples)

    # 4. Hidden Intent Logic (The true 'Brain' we want to learn)
    # y = weighted combination + noise
    # We want the model to learn that Preference + Quality + Position matters most.
    latent_score = (
        (user_category_score * 0.4) + 
        (video_quality * 0.3) + 
        (has_products * 0.2) - 
        (position_index * 0.05) + # Higher position index = less likely to click
        np.random.normal(0, 0.1, n_samples)
    )

    # 5. Labels (Ground Truth)
    # Click (is_active_engagement)
    is_click = (latent_score > 1.5).astype(int)
    
    # High Intent (Add to Cart / Purchase)
    is_purchase = (is_click & (np.random.uniform(0, 1, n_samples) > 0.7)).astype(int)

    # 6. Build DataFrame
    df = pd.DataFrame({
        "sample_id": range(n_samples),
        "position_index": position_index,
        "user_category_score": user_category_score,
        "video_quality": video_quality,
        "has_products": has_products,
        "hour_of_day": hour_of_day,
        "completion_rate": completion_rate,
        "is_click": is_click,
        "is_purchase": is_purchase
    })

    # Ensure directories exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Save to Parquet (standard ML format)
    out_path = "data/processed/synthetic_training.parquet"
    df.to_parquet(out_path, index=False)
    
    print(f"✅ Generated {n_samples} samples.")
    print(f"✅ Saved to: {out_path}")
    print(f"📊 Summary:")
    print(f"   Click Rate: {df['is_click'].mean():.2%}")
    print(f"   Purchase Rate: {df['is_purchase'].mean():.2%}")

if __name__ == "__main__":
    generate_synthetic_data()
