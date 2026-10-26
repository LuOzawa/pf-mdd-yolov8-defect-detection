from __future__ import annotations

"""
PF-MDD pipeline verification script.

Checks that the current workspace contains a trainable PF-MDD YOLO dataset with
valid paths, labels, and supporting metadata.
"""

import json
import sys
from pathlib import Path

import yaml

from project_config import (
    DATA_DIR,
    YOLO_CLASS_NAMES,
    dataset_has_required_layout,
    ensure_dataset_yaml,
    preferred_training_dataset_dir,
)

sys.stdout.reconfigure(encoding="utf-8")


def load_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"  [ERROR] Failed to load {path.name}: {exc}")
        return None


def verify_dataset_info(issues: list[str]) -> None:
    print("\n[STEP 1] Dataset Metadata")
    print("-" * 70)
    dataset_info = DATA_DIR / "pf_mdd_dataset_info.json"
    if not dataset_info.exists():
        issues.append("Missing data/pf_mdd_dataset_info.json")
        print(f"  [MISSING] {dataset_info}")
        return

    data = load_json(dataset_info)
    if not data:
        issues.append("Could not parse pf_mdd_dataset_info.json")
        return

    summary = data.get("summary", {})
    print(f"  [OK] Dataset info found: {dataset_info}")
    print(f"  - Total samples: {summary.get('total_samples', 'n/a')}")
    print(f"  - Defect samples: {summary.get('defect_samples', 'n/a')}")
    print(f"  - No-defect samples: {summary.get('nodefect_samples', 'n/a')}")
    print(f"  - Defect classes: {summary.get('num_defect_types', 'n/a')}")


def verify_conversion_stats(issues: list[str]) -> None:
    print("\n[STEP 2] Label Conversion")
    print("-" * 70)
    stats_path = DATA_DIR / "yolo_labels" / "conversion_stats.json"
    if not stats_path.exists():
        issues.append("Missing data/yolo_labels/conversion_stats.json")
        print(f"  [MISSING] {stats_path}")
        return

    data = load_json(stats_path)
    if not data:
        issues.append("Could not parse conversion_stats.json")
        return

    print(f"  [OK] Conversion stats found: {stats_path}")
    print(f"  - Processed samples: {data.get('total_processed', 0)}")
    print(f"  - Successful conversions: {data.get('successful', 0)}")
    print(f"  - Total boxes: {data.get('total_boxes', 0)}")
    if "class_mapping" in data:
        print(f"  - Class mapping entries: {len(data['class_mapping'])}")


def verify_yaml_and_labels(issues: list[str], dataset_dir: Path) -> None:
    print("\n[STEP 3] YOLO Dataset")
    print("-" * 70)
    if not dataset_has_required_layout(dataset_dir):
        issues.append(f"Dataset layout is incomplete: {dataset_dir}")
        print(f"  [MISSING] Expected images/labels splits under {dataset_dir}")
        return
    yaml_path = ensure_dataset_yaml(dataset_dir)

    with open(yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    expected_nc = len(YOLO_CLASS_NAMES)
    if int(data.get("nc", -1)) != expected_nc:
        issues.append(f"pf_mdd.yaml has nc={data.get('nc')} but expected {expected_nc}")

    path_value = Path(str(data.get("path", "")))
    if path_value.resolve() != dataset_dir.resolve():
        issues.append("pf_mdd.yaml path does not match the current workspace dataset directory")

    print(f"  [OK] YAML found: {yaml_path}")
    print(f"  - nc: {data.get('nc')}")
    print(f"  - path: {data.get('path')}")

    invalid_labels: list[str] = []
    total_images = 0
    total_labels = 0
    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_count = len(list(image_dir.glob("*.png")))
        label_count = len(list(label_dir.glob("*.txt")))
        total_images += image_count
        total_labels += label_count
        print(f"  - {split}: {image_count} images, {label_count} labels")

        for label_path in label_dir.glob("*.txt"):
            with open(label_path, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    class_id = int(line.split()[0])
                    if class_id < 0 or class_id >= expected_nc:
                        invalid_labels.append(f"{label_path.name}:{line_no} -> {class_id}")
                        if len(invalid_labels) >= 10:
                            break
            if len(invalid_labels) >= 10:
                break

    if total_images == 0 or total_labels == 0:
        issues.append("YOLO dataset appears empty")

    if invalid_labels:
        issues.append("Found out-of-range YOLO class IDs: " + ", ".join(invalid_labels))
    else:
        print("  [OK] Label IDs are within range")


def verify_split_statistics(issues: list[str], dataset_dir: Path) -> None:
    print("\n[STEP 4] Split Statistics")
    print("-" * 70)
    stats_path = dataset_dir / "split_statistics.json"
    if not stats_path.exists():
        print(f"  [SKIP] Split statistics not found at {stats_path}")
        return

    data = load_json(stats_path)
    if not data:
        issues.append("Could not parse split_statistics.json")
        return

    print(f"  [OK] Split statistics found: {stats_path}")
    for split in ("train", "val", "test"):
        split_data = data.get(split, {})
        if isinstance(split_data, dict):
            print(f"  - {split}: {split_data.get('count', 'n/a')} samples")


def main() -> int:
    print("=" * 70)
    print("PF-MDD PIPELINE VERIFICATION")
    print("=" * 70)
    dataset_dir = preferred_training_dataset_dir()
    print(f"[INFO] Active dataset directory: {dataset_dir}")

    issues: list[str] = []
    verify_dataset_info(issues)
    verify_conversion_stats(issues)
    verify_yaml_and_labels(issues, dataset_dir)
    verify_split_statistics(issues, dataset_dir)

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    if issues:
        print(f"[WARNING] Found {len(issues)} issue(s):")
        for idx, issue in enumerate(issues, start=1):
            print(f"  {idx}. {issue}")
        return 1

    print("[SUCCESS] Dataset and configuration look ready for YOLOv8 training.")
    print("Run `python train_baseline.py` to start training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
