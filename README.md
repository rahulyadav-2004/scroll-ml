# Scroll ML

Recommendation and Commerce Intelligence Machine Learning Engine.

## Setup & Connection

1. **Virtual Environment**: `python3 -m venv venv && source venv/bin/activate`
2. **Dependencies**: `pip install -r requirements.txt`
3. **Database**: Credentials are stored in `.env`.
4. **Verification**: Run `python src/test_db_conn.py` to verify connection.

## 🚀 Pipeline Operations

- **Train (Synthetic)**: `python src/train_hybrid_model.py` (Validates the machine).
- **Inference**: `python src/inference_service.py` (Starts the FastAPI server on port 8001).
- **Collect**: `python src/dataset_builder.py` (Pulls real signals from Postgres → Parquet).
- **Evaluate**: `python src/shadow_evaluator.py` (Compares ML vs Heuristic accuracy).
