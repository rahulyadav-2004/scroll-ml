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

## Automated Production Training

The inference service can now retrain itself on a separate ML instance.

What it does:
- builds a fresh real dataset from Postgres
- trains candidate engagement and commerce models into versioned artifact folders
- writes training status + manifest files
- only promotes candidate models to active paths when promotion rules pass
- hot-reloads the promoted models in the running inference process

### Environment Variables

Set these on the ML instance:

```bash
AUTO_TRAIN_ENABLED=true
AUTO_TRAIN_INTERVAL_SECONDS=21600
AUTO_TRAIN_INITIAL_DELAY_SECONDS=120
AUTO_TRAIN_DATASET_LIMIT=50000
AUTO_TRAIN_MIN_ROWS=250
AUTO_TRAIN_MIN_CLICK_POSITIVES=12
AUTO_TRAIN_MIN_COMMERCE_POSITIVES=12
AUTO_TRAIN_ALLOW_SYNTHETIC_FALLBACK_PROMOTION=false
AUTO_TRAIN_RETAIN_ARTIFACT_RUNS=10
ML_ADMIN_TOKEN=change-this
```

### Runtime Endpoints

- `GET /health`
  Returns model load status and the latest training status.
- `GET /training/status`
  Returns scheduler config, latest training result, and the active manifest.
- `POST /admin/retrain`
  Starts a manual background retrain. Send header `X-ML-Admin-Token: <ML_ADMIN_TOKEN>`.

Optional query param:

- `force_promote_fallback=true`
  Allows promoting fallback-trained models even if real data is still sparse.
  Keep this `false` in production unless you intentionally want bootstrap behavior.

### Manual Full Training Cycle

You can also run the same production pipeline once from the CLI:

```bash
python src/training_pipeline.py
```

This writes:
- `models/training_status.json`
- `models/active_manifest.json`
- `models/artifacts/<run_id>/...`
