import json
import os
import shutil
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from .dataset_builder import build_training_dataset
    from .train_hybrid_model import train_v1_shadow_with_metadata
    from .train_product_intent import train_product_intent_with_metadata
except ImportError:
    from dataset_builder import build_training_dataset
    from train_hybrid_model import train_v1_shadow_with_metadata
    from train_product_intent import train_product_intent_with_metadata


MODELS_DIR = Path("models")
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
ACTIVE_ENGAGEMENT_PATH = MODELS_DIR / "v1_shadow.joblib"
ACTIVE_COMMERCE_PATH = MODELS_DIR / "commerce_v1.joblib"
ACTIVE_MANIFEST_PATH = MODELS_DIR / "active_manifest.json"
TRAINING_STATUS_PATH = MODELS_DIR / "training_status.json"


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, path)


def _read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _copy_atomic(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dst)


@dataclass
class AutoTrainingConfig:
    enabled: bool = False
    interval_seconds: int = 21600
    initial_delay_seconds: int = 120
    dataset_limit: int = 50000
    min_rows: int = 250
    min_click_positives: int = 12
    min_commerce_positives: int = 12
    allow_synthetic_fallback_promotion: bool = False
    retain_artifact_runs: int = 10

    @classmethod
    def from_env(cls):
        return cls(
            enabled=_env_bool("AUTO_TRAIN_ENABLED", False),
            interval_seconds=int(os.getenv("AUTO_TRAIN_INTERVAL_SECONDS", "21600")),
            initial_delay_seconds=int(os.getenv("AUTO_TRAIN_INITIAL_DELAY_SECONDS", "120")),
            dataset_limit=int(os.getenv("AUTO_TRAIN_DATASET_LIMIT", "50000")),
            min_rows=int(os.getenv("AUTO_TRAIN_MIN_ROWS", "250")),
            min_click_positives=int(os.getenv("AUTO_TRAIN_MIN_CLICK_POSITIVES", "12")),
            min_commerce_positives=int(os.getenv("AUTO_TRAIN_MIN_COMMERCE_POSITIVES", "12")),
            allow_synthetic_fallback_promotion=_env_bool(
                "AUTO_TRAIN_ALLOW_SYNTHETIC_FALLBACK_PROMOTION",
                False,
            ),
            retain_artifact_runs=int(os.getenv("AUTO_TRAIN_RETAIN_ARTIFACT_RUNS", "10")),
        )


def build_dataset_summary(df):
    purchase = df.get("is_purchase", pd.Series(dtype=int)).fillna(0).astype(int)
    add_to_cart = df.get("is_add_to_cart", pd.Series(dtype=int)).fillna(0).astype(int)
    product_click = df.get("is_product_click", pd.Series(dtype=int)).fillna(0).astype(int)
    click = df.get("is_click", pd.Series(dtype=int)).fillna(0).astype(int)
    commerce_intent = ((purchase == 1) | (add_to_cart == 1) | (product_click == 1)).astype(int)

    return {
        "rows": int(len(df)),
        "click_positives": int((click == 1).sum()),
        "purchase_positives": int((purchase == 1).sum()),
        "add_to_cart_positives": int((add_to_cart == 1).sum()),
        "product_click_positives": int((product_click == 1).sum()),
        "commerce_positives": int((commerce_intent == 1).sum()),
        "click_rate": float(click.mean()) if len(df) else 0.0,
        "commerce_rate": float(commerce_intent.mean()) if len(df) else 0.0,
    }


def read_training_status():
    return _read_json(TRAINING_STATUS_PATH, default={})


def read_active_manifest():
    return _read_json(ACTIVE_MANIFEST_PATH, default={})


def _cleanup_old_artifacts(retain_count):
    if retain_count <= 0 or not ARTIFACTS_DIR.exists():
        return

    artifact_dirs = sorted(
        [path for path in ARTIFACTS_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )

    for old_path in artifact_dirs[retain_count:]:
        shutil.rmtree(old_path, ignore_errors=True)


def _should_promote(config, dataset_summary, engagement_meta, commerce_meta, force_promote_fallback):
    reasons = []
    active_models_exist = ACTIVE_ENGAGEMENT_PATH.exists() and ACTIVE_COMMERCE_PATH.exists()

    if dataset_summary["rows"] < config.min_rows:
        reasons.append(
            f"dataset too small ({dataset_summary['rows']} rows < {config.min_rows})"
        )

    if dataset_summary["click_positives"] < config.min_click_positives:
        reasons.append(
            f"not enough real click positives ({dataset_summary['click_positives']} < {config.min_click_positives})"
        )

    if dataset_summary["commerce_positives"] < config.min_commerce_positives:
        reasons.append(
            f"not enough real commerce positives ({dataset_summary['commerce_positives']} < {config.min_commerce_positives})"
        )

    if not force_promote_fallback and not config.allow_synthetic_fallback_promotion and active_models_exist:
        if engagement_meta.get("used_fallback_dataset"):
            reasons.append("engagement training used synthetic fallback")
        if commerce_meta.get("used_fallback_dataset"):
            reasons.append("commerce training used synthetic fallback")

    if not active_models_exist:
        return True, reasons

    if force_promote_fallback:
        return True, reasons

    return len(reasons) == 0, reasons


def run_training_cycle(config=None, force_promote_fallback=False, trigger="manual"):
    config = config or AutoTrainingConfig.from_env()
    started_at = _utc_now_iso()

    _write_json_atomic(
        TRAINING_STATUS_PATH,
        {
            "state": "running",
            "trigger": trigger,
            "started_at": started_at,
            "config": asdict(config),
        },
    )

    try:
        df = build_training_dataset(limit=config.dataset_limit)
        if df is None or df.empty:
            raise RuntimeError("dataset build returned no usable rows")

        dataset_summary = build_dataset_summary(df)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = ARTIFACTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        engagement_output_path = run_dir / "v1_shadow.joblib"
        commerce_output_path = run_dir / "commerce_v1.joblib"

        _, engagement_meta = train_v1_shadow_with_metadata(
            output_path=engagement_output_path
        )
        _, commerce_meta = train_product_intent_with_metadata(
            output_path=commerce_output_path
        )

        promotion_allowed, promotion_reasons = _should_promote(
            config,
            dataset_summary,
            engagement_meta,
            commerce_meta,
            force_promote_fallback=force_promote_fallback,
        )

        run_manifest = {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": _utc_now_iso(),
            "trigger": trigger,
            "config": asdict(config),
            "dataset_summary": dataset_summary,
            "engagement": engagement_meta,
            "commerce": commerce_meta,
            "promotion_allowed": promotion_allowed,
            "promotion_reasons": promotion_reasons,
        }
        _write_json_atomic(run_dir / "manifest.json", run_manifest)

        promoted = False
        if promotion_allowed:
            _copy_atomic(engagement_output_path, ACTIVE_ENGAGEMENT_PATH)
            _copy_atomic(commerce_output_path, ACTIVE_COMMERCE_PATH)
            _write_json_atomic(ACTIVE_MANIFEST_PATH, run_manifest)
            promoted = True

        _cleanup_old_artifacts(config.retain_artifact_runs)

        result = {
            "state": "completed",
            "trigger": trigger,
            "started_at": started_at,
            "completed_at": _utc_now_iso(),
            "promoted": promoted,
            "promotion_reasons": promotion_reasons,
            "run_id": run_id,
            "dataset_summary": dataset_summary,
            "engagement": engagement_meta,
            "commerce": commerce_meta,
        }
        _write_json_atomic(TRAINING_STATUS_PATH, result)
        return result

    except Exception as exc:
        result = {
            "state": "failed",
            "trigger": trigger,
            "started_at": started_at,
            "completed_at": _utc_now_iso(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "config": asdict(config),
        }
        _write_json_atomic(TRAINING_STATUS_PATH, result)
        return result


if __name__ == "__main__":
    output = run_training_cycle()
    print(json.dumps(output, indent=2, sort_keys=True))
