"""Train all churn models sequentially and persist artifacts.

Usage:
    python scripts/train_all_models.py
    python scripts/train_all_models.py --sample-size 50000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import init_db, log_training_result
from src.models import MODEL_REGISTRY, load_model_artifact, save_model_artifact, train_model
from src.preprocessing import load_and_prepare_data, save_preprocessing_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Train all customer churn models")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Max rows to use from train.csv (default: 50000)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training for models that already have saved artifacts",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    init_db()

    print("Loading and preprocessing training data...")
    X_train, X_test, y_train, y_test, label_encoders, scaler, _ = load_and_prepare_data(
        sample_size=args.sample_size
    )

    preprocessor_path = artifacts_dir / "preprocessor.joblib"
    save_preprocessing_artifacts(label_encoders, scaler, preprocessor_path)
    print(f"Saved preprocessing artifacts to: {preprocessor_path}")

    leaderboard = []

    for idx, model_name in enumerate(MODEL_REGISTRY.keys(), start=1):
        print(f"[{idx}/{len(MODEL_REGISTRY)}] Training {model_name}...")

        if args.skip_existing and load_model_artifact(model_name, artifacts_dir) is not None:
            print(f"  - Skipped (artifact already exists): {model_name}")
            continue

        model, metrics, _ = train_model(model_name, X_train, X_test, y_train, y_test)

        if model is None or (isinstance(metrics, dict) and "error" in metrics):
            error_msg = metrics.get("error", "unknown error") if isinstance(metrics, dict) else "unknown error"
            print(f"  - Failed: {error_msg}")
            log_training_result(model_name, None, status=f"failed: {error_msg}")
            continue

        model_path = save_model_artifact(model, model_name, artifacts_dir)
        print(f"  - Saved model artifact: {model_path}")
        log_training_result(model_name, metrics, status="success")

        leaderboard.append(
            {
                "Model": model_name,
                "Category": MODEL_REGISTRY[model_name]["category"],
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1 Score"],
                "ArtifactPath": str(model_path),
            }
        )

    if leaderboard:
        df = pd.DataFrame(leaderboard).sort_values("F1 Score", ascending=False)
        csv_path = artifacts_dir / "leaderboard.csv"
        json_path = artifacts_dir / "leaderboard.json"
        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)

        print("\nTraining complete.")
        print(f"Leaderboard CSV: {csv_path}")
        print(f"Leaderboard JSON: {json_path}")
        print("Top 5 models by F1:")
        print(df[["Model", "F1 Score"]].head(5).to_string(index=False))
    else:
        print("No models were successfully trained.")


if __name__ == "__main__":
    main()
