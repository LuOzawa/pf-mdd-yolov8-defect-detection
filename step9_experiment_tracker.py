from __future__ import annotations

"""
Track PF-MDD experiments:
- Optional: run training
- Evaluate selected run
- Append overall/per-class metrics to CSV for cross-run comparison
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from evaluate_model import ModelEvaluator, resolve_run_dataset_yaml
from train_baseline import train_baseline_yolov8
from project_config import TRAINING_RESULTS_DIR

sys.stdout.reconfigure(encoding="utf-8")


SUMMARY_CSV = TRAINING_RESULTS_DIR / "experiment_tracking_summary.csv"
PER_CLASS_CSV = TRAINING_RESULTS_DIR / "experiment_tracking_per_class.csv"


def latest_complete_run() -> Path:
    runs = sorted(path for path in TRAINING_RESULTS_DIR.iterdir() if path.is_dir())
    valid = [run for run in runs if (run / "weights" / "best.pt").exists()]
    if not valid:
        raise FileNotFoundError("No complete training run found under training_results/.")
    return valid[-1]


def ensure_summary_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tracked_at",
                "run_name",
                "dataset_yaml",
                "mAP50",
                "mAP50_95",
                "precision",
                "recall",
                "fps",
                "total_inference_ms",
            ]
        )


def ensure_per_class_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tracked_at",
                "run_name",
                "class_name",
                "instances",
                "precision",
                "recall",
                "ap50",
                "ap50_95",
            ]
        )


def append_tracking(run_dir: Path, dataset_yaml: Path, eval_result: dict) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ensure_summary_header(SUMMARY_CSV)
    ensure_per_class_header(PER_CLASS_CSV)

    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                now,
                run_dir.name,
                str(dataset_yaml),
                f"{eval_result.get('mAP50', 0.0):.6f}",
                f"{eval_result.get('mAP5095', 0.0):.6f}",
                f"{eval_result.get('precision', 0.0):.6f}",
                f"{eval_result.get('recall', 0.0):.6f}",
                f"{eval_result.get('fps', 0.0):.6f}",
                f"{eval_result.get('total_inference_ms', 0.0):.6f}",
            ]
        )

    classes = sorted(eval_result.get("per_class_AP50", {}).keys())
    with open(PER_CLASS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for cls_name in classes:
            writer.writerow(
                [
                    now,
                    run_dir.name,
                    cls_name,
                    int(eval_result.get("per_class_instances", {}).get(cls_name, 0)),
                    f"{eval_result.get('per_class_precision', {}).get(cls_name, 0.0):.6f}",
                    f"{eval_result.get('per_class_recall', {}).get(cls_name, 0.0):.6f}",
                    f"{eval_result.get('per_class_AP50', {}).get(cls_name, 0.0):.6f}",
                    f"{eval_result.get('per_class_AP5095', {}).get(cls_name, 0.0):.6f}",
                ]
            )

    print(f"[SAVED] {SUMMARY_CSV}")
    print(f"[SAVED] {PER_CLASS_CSV}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PF-MDD experiment tracker")
    parser.add_argument("--train", action="store_true", help="Run training before evaluation")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--workers", type=int, default=None, help="Override workers")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    parser.add_argument("--fraction", type=float, default=None, help="Override data fraction for quick experiments")
    parser.add_argument("--model-weights", default="yolov8s.pt", help="Model weights for training")
    parser.add_argument(
        "--overrides-json",
        default=None,
        help="Raw JSON string for extra train_baseline_yolov8 overrides",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Evaluate a specific run directory (skip training unless --train)",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.workers is not None:
        overrides["workers"] = args.workers
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.fraction is not None:
        overrides["fraction"] = args.fraction
    if args.overrides_json:
        raw = json.loads(args.overrides_json)
        if not isinstance(raw, dict):
            raise ValueError("--overrides-json must decode to a JSON object")
        overrides.update(raw)
    return overrides


def main() -> int:
    args = parse_args()
    run_dir: Path | None = Path(args.run_dir) if args.run_dir else None

    if args.train:
        print("[STEP] Training before evaluation...")
        overrides = build_overrides(args)
        results, _, _ = train_baseline_yolov8(overrides=overrides, model_weights=args.model_weights)
        if results is None:
            print("[ERROR] Training failed, skip tracking.")
            return 1
        run_dir = Path(results.save_dir)
        print(f"[INFO] New run: {run_dir}")

    if run_dir is None:
        run_dir = latest_complete_run()
        print(f"[INFO] Using latest complete run: {run_dir}")

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[ERROR] Missing best.pt: {best_pt}")
        return 1

    evaluator = ModelEvaluator(TRAINING_RESULTS_DIR)
    dataset_yaml = resolve_run_dataset_yaml(run_dir)
    print(f"[INFO] Dataset YAML: {dataset_yaml}")
    eval_result = evaluator.evaluate_test_set(best_pt, dataset_yaml)
    if eval_result is None:
        return 1

    append_tracking(run_dir=run_dir, dataset_yaml=dataset_yaml, eval_result=eval_result)
    print("[DONE] Experiment tracking updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
