#!/usr/bin/env python
#!/usr/bin/env python
from __future__ import annotations

"""
PF-MDD model evaluation utilities.

This module evaluates a trained YOLOv8 detector on the PF-MDD test split and
produces plots/reports using only metrics returned by Ultralytics.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from project_config import TRAINING_REPORTS_DIR, TRAINING_RESULTS_DIR, ensure_dataset_yaml, preferred_training_dataset_dir

sys.stdout.reconfigure(encoding="utf-8")


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
DEFAULT_MIN_STABLE_INSTANCES = 5


def resolve_run_dataset_yaml(result_dir: Path) -> Path:
    """Resolve dataset YAML used by a training run and remap stale absolute paths to the current workspace."""
    args_yaml = result_dir / "args.yaml"
    candidate_paths: list[Path] = []

    if args_yaml.exists():
        try:
            with open(args_yaml, "r", encoding="utf-8") as handle:
                args_data = yaml.safe_load(handle) or {}
            data_value = args_data.get("data")
            if data_value:
                raw = Path(str(data_value))
                if raw.suffix.lower() == ".yaml":
                    candidate_paths.append(raw)
                else:
                    candidate_paths.append(raw / "pf_mdd.yaml")
        except Exception:
            pass

    local_base = Path(__file__).resolve().parent
    for candidate in candidate_paths:
        if candidate.exists():
            return ensure_dataset_yaml(candidate.parent, candidate)

        # Map stale absolute paths such as C:/.../code/<dataset>/pf_mdd.yaml into current workspace.
        parent_name = candidate.parent.name
        if parent_name:
            mapped = local_base / parent_name / "pf_mdd.yaml"
            if mapped.exists() or mapped.parent.exists():
                return ensure_dataset_yaml(mapped.parent, mapped)

    fallback_dir = preferred_training_dataset_dir()
    return ensure_dataset_yaml(fallback_dir)


class ModelEvaluator:
    """Evaluate PF-MDD training runs and export plots/reports."""

    def __init__(self, results_dir: str | Path | None = None, min_stable_instances: int = DEFAULT_MIN_STABLE_INSTANCES):
        self.results_dir = Path(results_dir) if results_dir else None
        self.min_stable_instances = max(1, int(min_stable_instances))
        self.evaluation_results: dict = {}
        self.confusion_matrix = None
        self.confusion_labels: list[str] = []

    def _apply_class_stability_summary(self, evaluation: dict) -> None:
        """Annotate evaluation with stable/low-sample/no-sample class groups."""
        instances_map: dict[str, int] = evaluation.get("per_class_instances", {})
        stable_classes = sorted([name for name, n in instances_map.items() if int(n) >= self.min_stable_instances])
        low_sample_classes = sorted([name for name, n in instances_map.items() if 0 < int(n) < self.min_stable_instances])
        zero_sample_classes = sorted([name for name, n in instances_map.items() if int(n) == 0])

        evaluation["min_stable_instances"] = self.min_stable_instances
        evaluation["stable_classes"] = stable_classes
        evaluation["low_sample_classes"] = low_sample_classes
        evaluation["zero_sample_classes"] = zero_sample_classes
        evaluation["stable_class_count"] = len(stable_classes)

        if not stable_classes:
            evaluation["stable_macro_precision"] = None
            evaluation["stable_macro_recall"] = None
            evaluation["stable_macro_ap50"] = None
            evaluation["stable_macro_ap5095"] = None
            return

        def mean_metric(metric_key: str) -> float:
            values = [float(evaluation.get(metric_key, {}).get(name, 0.0)) for name in stable_classes]
            return float(np.mean(values)) if values else 0.0

        evaluation["stable_macro_precision"] = mean_metric("per_class_precision")
        evaluation["stable_macro_recall"] = mean_metric("per_class_recall")
        evaluation["stable_macro_ap50"] = mean_metric("per_class_AP50")
        evaluation["stable_macro_ap5095"] = mean_metric("per_class_AP5095")

    def load_training_results(self, results_path: str | Path) -> dict | list:
        """Load training results from CSV or JSON."""
        results_file = Path(results_path)
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        if results_file.suffix == ".csv":
            import pandas as pd

            return pd.read_csv(results_file).to_dict("records")

        if results_file.suffix == ".json":
            with open(results_file, "r", encoding="utf-8") as handle:
                return json.load(handle)

        raise ValueError(f"Unsupported results format: {results_file.suffix}")

    def evaluate_test_set(self, model_path: str | Path, test_data_yaml: str | Path) -> dict | None:
        """Evaluate a trained model on the PF-MDD test split."""
        from ultralytics import YOLO

        print("\n" + "=" * 70)
        print("[EVALUATION] Testing on PF-MDD Test Set")
        print("=" * 70)

        model_path = Path(model_path)
        test_data_yaml = Path(test_data_yaml)

        try:
            print(f"\n[STEP 1] Loading model: {model_path}")
            model = YOLO(str(model_path))

            print("[STEP 2] Evaluating on test set...")
            val_kwargs = dict(
                data=str(test_data_yaml),
                split="test",
                conf=0.001,
                iou=0.6,
                max_det=300,
                verbose=True,
                device="cpu",
                workers=0,
                plots=False,
            )
            try:
                metrics = model.val(**val_kwargs)
            except PermissionError:
                # Some restricted Windows environments block ultralytics' default ThreadPool backend.
                from concurrent.futures import ThreadPoolExecutor
                import ultralytics.data.dataset as yolo_dataset_module

                class SafeThreadPool:
                    """ThreadPool fallback that avoids multiprocessing pipes."""

                    def __init__(self, processes=None, *args, **kwargs):
                        max_workers = max(1, int(processes or 1))
                        self._executor = ThreadPoolExecutor(max_workers=max_workers)

                    def imap(self, func, iterable):
                        futures = [self._executor.submit(func, item) for item in iterable]
                        for future in futures:
                            yield future.result()

                    def __enter__(self):
                        return self

                    def __exit__(self, exc_type, exc, tb):
                        self._executor.shutdown(wait=True)

                print("[WARNING] Permission error with default data loader backend. Retrying with safe thread pool...")
                yolo_dataset_module.ThreadPool = SafeThreadPool
                metrics = model.val(**val_kwargs)

            speed = metrics.speed if isinstance(metrics.speed, dict) else {}
            evaluation = {
                "mAP50": float(metrics.box.map50),
                "mAP5095": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "preprocess_time": float(speed.get("preprocess", 0.0)),
                "inference_time": float(speed.get("inference", 0.0)),
                "postprocess_time": float(speed.get("postprocess", 0.0)),
                "per_class_AP50": {},
                "per_class_AP5095": {},
                "per_class_precision": {},
                "per_class_recall": {},
                "per_class_instances": {},
            }

            names = {int(class_id): name for class_id, name in metrics.names.items()}
            for class_id, name in names.items():
                evaluation["per_class_AP50"][name] = 0.0
                evaluation["per_class_AP5095"][name] = 0.0
                evaluation["per_class_precision"][name] = 0.0
                evaluation["per_class_recall"][name] = 0.0
                instances = 0
                if getattr(metrics, "nt_per_class", None) is not None and len(metrics.nt_per_class) > class_id:
                    instances = int(metrics.nt_per_class[class_id])
                evaluation["per_class_instances"][name] = instances

            for result_idx, class_id in enumerate(metrics.ap_class_index):
                name = names[int(class_id)]
                precision, recall, ap50, ap5095 = metrics.class_result(result_idx)
                evaluation["per_class_AP50"][name] = float(ap50)
                evaluation["per_class_AP5095"][name] = float(ap5095)
                evaluation["per_class_precision"][name] = float(precision)
                evaluation["per_class_recall"][name] = float(recall)

            total_time = (
                evaluation["preprocess_time"]
                + evaluation["inference_time"]
                + evaluation["postprocess_time"]
            )
            evaluation["total_inference_ms"] = total_time
            evaluation["fps"] = 1000.0 / total_time if total_time > 0 else 0.0
            self._apply_class_stability_summary(evaluation)

            self.evaluation_results = evaluation
            self.confusion_matrix = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
            self.confusion_labels = [names[idx] for idx in sorted(names)] + ["background"]

            print("\n[EVALUATION COMPLETE]")
            print("-" * 50)
            print(f"mAP@0.5:      {evaluation['mAP50']:.4f}")
            print(f"mAP@0.5:0.95: {evaluation['mAP5095']:.4f}")
            print(f"Precision:    {evaluation['precision']:.4f}")
            print(f"Recall:       {evaluation['recall']:.4f}")
            print(f"Inference:    {total_time:.1f}ms ({evaluation['fps']:.1f} FPS)")
            print(
                f"Stable classes (instances >= {self.min_stable_instances}): "
                f"{evaluation['stable_class_count']}/{len(evaluation['per_class_instances'])}"
            )
            if evaluation["stable_class_count"] > 0:
                print(
                    "Stable macro metrics - "
                    f"AP50: {evaluation['stable_macro_ap50']:.4f}, "
                    f"AP50-95: {evaluation['stable_macro_ap5095']:.4f}, "
                    f"P: {evaluation['stable_macro_precision']:.4f}, "
                    f"R: {evaluation['stable_macro_recall']:.4f}"
                )

            return evaluation

        except Exception as exc:
            print(f"[ERROR] Evaluation failed: {exc}")
            import traceback

            traceback.print_exc()
            return None

    def plot_training_curves(self, results_csv_path: str | Path, output_dir: str | Path | None = None) -> str | None:
        """Generate a compact training summary figure from Ultralytics results.csv."""
        import pandas as pd

        print("\n[VISUALIZATION] Generating training curves...")
        results_file = Path(results_csv_path)
        if not results_file.exists():
            print(f"[ERROR] Results file not found: {results_file}")
            return None

        df = pd.read_csv(results_file)
        output_dir = Path(output_dir) if output_dir else results_file.parent.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(20, 16))

        ax1 = fig.add_subplot(2, 3, 1)
        if {"train/box_loss", "train/cls_loss", "train/dfl_loss"}.issubset(df.columns):
            ax1.plot(df["epoch"], df["train/box_loss"], label="Box Loss", linewidth=2, color="#1565C0")
            ax1.plot(df["epoch"], df["train/cls_loss"], label="Cls Loss", linewidth=2, color="#EF6C00")
            ax1.plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss", linewidth=2, color="#C62828")
            ax1.set_title("Training Loss Curves", fontweight="bold")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()

        ax2 = fig.add_subplot(2, 3, 2)
        if {"val/box_loss", "val/cls_loss", "val/dfl_loss"}.issubset(df.columns):
            ax2.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", linewidth=2, linestyle="--", color="#1565C0")
            ax2.plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss", linewidth=2, linestyle="--", color="#EF6C00")
            ax2.plot(df["epoch"], df["val/dfl_loss"], label="Val DFL Loss", linewidth=2, linestyle="--", color="#C62828")
            ax2.set_title("Validation Loss Curves", fontweight="bold")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.legend()

        ax3 = fig.add_subplot(2, 3, 3)
        if {"metrics/mAP50(B)", "metrics/mAP50-95(B)"}.issubset(df.columns):
            ax3.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", linewidth=2.5, color="#2E7D32")
            ax3.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", linewidth=2.5, color="#6A1B9A")
            ax3.set_title("Mean Average Precision", fontweight="bold")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("mAP")
            ax3.legend()

        ax4 = fig.add_subplot(2, 3, 4)
        if {"metrics/precision(B)", "metrics/recall(B)"}.issubset(df.columns):
            ax4.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", linewidth=2, color="#D32F2F")
            ax4.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", linewidth=2, color="#1976D2")
            ax4.set_title("Precision and Recall", fontweight="bold")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Score")
            ax4.set_ylim([0, 1])
            ax4.legend()

        ax5 = fig.add_subplot(2, 3, 5)
        if self.evaluation_results:
            classes = list(self.evaluation_results["per_class_AP50"].keys())
            ap_values = [self.evaluation_results["per_class_AP50"][name] for name in classes]
            colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
            ax5.bar(range(len(classes)), ap_values, color=colors, edgecolor="black", linewidth=0.5)
            ax5.set_xticks(range(len(classes)))
            ax5.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
            ax5.set_xlabel("Defect Class")
            ax5.set_ylabel("AP@0.5")
            ax5.set_title("Per-Class AP@0.5", fontweight="bold")

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis("off")
        summary_lines = ["Training Summary", "=" * 30]
        if self.evaluation_results:
            summary_lines.extend(
                [
                    f"Final mAP@0.5:     {self.evaluation_results.get('mAP50', 0):.4f}",
                    f"Final mAP@0.5:0.95: {self.evaluation_results.get('mAP5095', 0):.4f}",
                    f"Precision:          {self.evaluation_results.get('precision', 0):.4f}",
                    f"Recall:             {self.evaluation_results.get('recall', 0):.4f}",
                    f"FPS:                {self.evaluation_results.get('fps', 0):.1f}",
                    f"Stable classes:     {self.evaluation_results.get('stable_class_count', 0)}",
                ]
            )
            stable_ap50 = self.evaluation_results.get("stable_macro_ap50")
            if stable_ap50 is not None:
                summary_lines.append(f"Stable macro AP@0.5:{stable_ap50:>9.4f}")
        if "epoch" in df.columns:
            summary_lines.extend(["=" * 30, f"Total Epochs:       {len(df)}"])
            if "metrics/mAP50(B)" in df.columns:
                best_row = int(df["metrics/mAP50(B)"].idxmax())
                summary_lines.append(f"Best Epoch:         {int(df['epoch'].iloc[best_row])}")

        ax6.text(
            0.1,
            0.9,
            "\n".join(summary_lines),
            transform=ax6.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        output_path = output_dir / "training_curves_comprehensive.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"[SAVED] Training curves: {output_path}")
        return str(output_path)

    def generate_confusion_matrix(self, output_dir: str | Path | None = None) -> str | None:
        """Plot the actual confusion matrix returned by Ultralytics."""
        if self.confusion_matrix is None:
            print("[WARNING] No confusion matrix available")
            return None

        print("\n[VISUALIZATION] Generating confusion matrix...")
        output_dir = Path(output_dir) if output_dir else TRAINING_REPORTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ultralytics stores detection confusion as [predicted, true].
        confusion = np.array(self.confusion_matrix, dtype=float).T
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion_pct = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums != 0) * 100
        labels = self.confusion_labels

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(confusion_pct, cmap="Blues", aspect="auto")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Prediction Percentage (%)", rotation=-90, va="bottom", fontsize=12)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted Class", fontweight="bold")
        ax.set_ylabel("True Class", fontweight="bold")
        ax.set_title("Confusion Matrix - PF-MDD Test Set", fontweight="bold")

        threshold = confusion_pct.max() / 2 if confusion_pct.size else 0
        for row_idx in range(confusion_pct.shape[0]):
            for col_idx in range(confusion_pct.shape[1]):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{confusion_pct[row_idx, col_idx]:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if confusion_pct[row_idx, col_idx] > threshold else "black",
                    fontsize=8,
                )

        plt.tight_layout()
        output_path = output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"[SAVED] Confusion matrix: {output_path}")
        return str(output_path)

    def generate_performance_report(self, output_path: str | Path | None = None) -> str | None:
        """Generate a markdown report using actual evaluation outputs."""
        if not self.evaluation_results:
            print("[WARNING] No evaluation results available")
            return None

        print("\n[REPORT] Generating performance report...")
        output_path = Path(output_path) if output_path else TRAINING_REPORTS_DIR / "baseline_performance_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        precision = self.evaluation_results.get("precision", 0.0)
        recall = self.evaluation_results.get("recall", 0.0)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        stable_ap50 = self.evaluation_results.get("stable_macro_ap50")
        stable_ap5095 = self.evaluation_results.get("stable_macro_ap5095")
        stable_precision = self.evaluation_results.get("stable_macro_precision")
        stable_recall = self.evaluation_results.get("stable_macro_recall")
        stable_threshold = int(self.evaluation_results.get("min_stable_instances", self.min_stable_instances))
        stable_class_count = int(self.evaluation_results.get("stable_class_count", 0))

        lines = [
            "# PF-MDD Baseline Model Performance Report",
            "",
            "## YOLOv8s on Plastic Film Multi-class Defect Detection Dataset",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            "This report presents the baseline performance of **YOLOv8s** trained on the **PF-MDD** dataset.",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| mAP@0.5 | {self.evaluation_results.get('mAP50', 0):.4f} |",
            f"| mAP@0.5:0.95 | {self.evaluation_results.get('mAP5095', 0):.4f} |",
            f"| Precision | {precision:.4f} |",
            f"| Recall | {recall:.4f} |",
            f"| F1-score | {f1:.4f} |",
            f"| Total inference time | {self.evaluation_results.get('total_inference_ms', 0):.2f} ms |",
            f"| FPS | {self.evaluation_results.get('fps', 0):.1f} |",
            f"| Stable classes threshold | >= {stable_threshold} instances |",
            f"| Stable class count | {stable_class_count} |",
            "",
            "",
            "### 1.1 Stable-Class Macro Metrics (more reliable summary)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]

        if stable_ap50 is None:
            lines.append("| Stable macro metrics | N/A (no class meets stable threshold) |")
        else:
            lines.extend(
                [
                    f"| Stable macro AP@0.5 | {stable_ap50:.4f} |",
                    f"| Stable macro AP@0.5:0.95 | {stable_ap5095:.4f} |",
                    f"| Stable macro Precision | {stable_precision:.4f} |",
                    f"| Stable macro Recall | {stable_recall:.4f} |",
                ]
            )

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 2. Per-Class Results")
        lines.append("")
        lines.append("| Class | Instances | Status | Precision | Recall | AP@0.5 | AP@0.5:0.95 |")
        lines.append("|-------|-----------|--------|-----------|--------|--------|-------------|")

        stable_classes = set(self.evaluation_results.get("stable_classes", []))
        low_sample_classes = set(self.evaluation_results.get("low_sample_classes", []))
        for class_name in sorted(self.evaluation_results["per_class_AP50"].keys()):
            if class_name in stable_classes:
                status = "stable"
            elif class_name in low_sample_classes:
                status = "low-sample"
            else:
                status = "no-sample"
            lines.append(
                "| {name} | {instances} | {status} | {precision:.4f} | {recall:.4f} | {ap50:.4f} | {ap5095:.4f} |".format(
                    name=class_name,
                    instances=self.evaluation_results["per_class_instances"].get(class_name, 0),
                    status=status,
                    precision=self.evaluation_results["per_class_precision"].get(class_name, 0.0),
                    recall=self.evaluation_results["per_class_recall"].get(class_name, 0.0),
                    ap50=self.evaluation_results["per_class_AP50"].get(class_name, 0.0),
                    ap5095=self.evaluation_results["per_class_AP5095"].get(class_name, 0.0),
                )
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Interpretation",
                "",
                "- Higher `mAP@0.5` indicates stronger coarse localization and classification performance.",
                "- `mAP@0.5:0.95` is stricter and better reflects localization quality on tiny defects.",
                f"- Classes with fewer than {stable_threshold} test instances are marked as `low-sample`; their metrics are unstable.",
                "- `no-sample` classes have no test instances in this run and should not be used for model ranking.",
                "- Prefer stable-class macro metrics for model-to-model comparison in long-tail settings.",
                "",
                "---",
                "",
                "## 4. Generated Artifacts",
                "",
                "- `training_curves_comprehensive.png`",
                "- `confusion_matrix.png`",
                "- `baseline_performance_report.md`",
                "",
                "---",
                "",
                "## 5. Next Steps",
                "",
                "- Improve minority-class coverage through targeted augmentation or new data collection.",
                "- Compare YOLOv8s against a smaller baseline such as YOLOv8n and a stronger model such as YOLOv8m.",
                "- Add negative-only film images into training/evaluation to measure false positive control.",
            ]
        )

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        print(f"[SAVED] Performance report: {output_path}")
        return str(output_path)


def main() -> int:
    """Evaluate the latest PF-MDD training run."""
    print("#" * 70)
    print("# PF-MDD MODEL EVALUATION SYSTEM")
    print("#" * 70)

    TRAINING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluator = ModelEvaluator(TRAINING_RESULTS_DIR)

    result_dirs = sorted(path for path in TRAINING_RESULTS_DIR.iterdir() if path.is_dir())
    if not result_dirs:
        print("[ERROR] No training results found.")
        print("Please run train_baseline.py first.")
        return 1

    valid_result_dirs = []
    skipped_result_dirs = []
    for result_dir in result_dirs:
        best_weights = result_dir / "weights" / "best.pt"
        if best_weights.exists():
            valid_result_dirs.append(result_dir)
        else:
            skipped_result_dirs.append(result_dir.name)

    if skipped_result_dirs:
        print(f"[WARNING] Skipping incomplete runs: {', '.join(skipped_result_dirs)}")

    if not valid_result_dirs:
        print("[ERROR] No complete training run found (missing weights/best.pt).")
        return 1

    latest_result = valid_result_dirs[-1]
    print(f"\n[INFO] Using results from: {latest_result.name}")

    model_path = latest_result / "weights" / "best.pt"
    results_csv = latest_result / "results.csv"
    data_yaml = resolve_run_dataset_yaml(latest_result)
    print(f"[INFO] Using dataset config: {data_yaml}")

    if not model_path.exists():
        print(f"[ERROR] Model weights not found: {model_path}")
        return 1

    eval_results = evaluator.evaluate_test_set(model_path, data_yaml)
    if eval_results is None:
        return 1

    if results_csv.exists():
        evaluator.plot_training_curves(results_csv, TRAINING_REPORTS_DIR)
    else:
        print("[WARNING] Results CSV not found, skipping curve generation")

    evaluator.generate_confusion_matrix(TRAINING_REPORTS_DIR)
    evaluator.generate_performance_report(TRAINING_REPORTS_DIR / "baseline_performance_report.md")

    print("\n" + "=" * 70)
    print("[EVALUATION COMPLETE]")
    print("=" * 70)
    print(f"Output directory: {TRAINING_REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
