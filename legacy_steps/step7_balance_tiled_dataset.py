from __future__ import annotations

"""
Class-balancing expansion for PF-MDD tiled training dataset.

Goal:
- Increase train-set positive data volume.
- Reduce class imbalance by augmenting minority classes to a target box count.
- Keep val/test unchanged to preserve fair evaluation.
"""

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from project_config import (
    BALANCED_TILED_DATASET_DIR,
    DEFECT_TYPE_TO_YOLO_INDEX,
    EXPANDED_TILED_DATASET_DIR,
    TILED_DATASET_DIR,
    YOLO_CLASS_NAMES,
    dataset_has_required_layout,
    ensure_dataset_yaml,
)

sys.stdout.reconfigure(encoding="utf-8")


def load_yolo_labels(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    labels: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        labels.append([float(cls_id), xc, yc, bw, bh])
    return labels


def save_yolo_labels(labels: list[list[float]], label_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as handle:
        for cls_id, xc, yc, bw, bh in labels:
            handle.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def transform_labels_hflip(labels: list[list[float]]) -> list[list[float]]:
    return [[cls_id, 1.0 - xc, yc, bw, bh] for cls_id, xc, yc, bw, bh in labels]


def transform_labels_vflip(labels: list[list[float]]) -> list[list[float]]:
    return [[cls_id, xc, 1.0 - yc, bw, bh] for cls_id, xc, yc, bw, bh in labels]


def transform_labels_rot90(labels: list[list[float]]) -> list[list[float]]:
    # For square images, w/h swap under 90-degree rotation.
    return [[cls_id, 1.0 - yc, xc, bh, bw] for cls_id, xc, yc, bw, bh in labels]


def transform_labels_rot180(labels: list[list[float]]) -> list[list[float]]:
    return [[cls_id, 1.0 - xc, 1.0 - yc, bw, bh] for cls_id, xc, yc, bw, bh in labels]


@dataclass
class BalanceSummary:
    source_dataset: str
    target_boxes_per_class: int
    initial_train_images: int
    final_train_images: int
    added_train_images: int
    initial_class_box_counts: dict[int, int]
    final_class_box_counts: dict[int, int]
    impossible_classes: list[int]

    def to_dict(self) -> dict:
        return {
            "source_dataset": self.source_dataset,
            "target_boxes_per_class": self.target_boxes_per_class,
            "initial_train_images": self.initial_train_images,
            "final_train_images": self.final_train_images,
            "added_train_images": self.added_train_images,
            "initial_class_box_counts": {str(k): v for k, v in sorted(self.initial_class_box_counts.items())},
            "final_class_box_counts": {str(k): v for k, v in sorted(self.final_class_box_counts.items())},
            "impossible_classes": self.impossible_classes,
        }


class TiledDatasetBalancer:
    def __init__(self, source_dir: Path, output_dir: Path, target_boxes_per_class: int | None = None):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.target_boxes_per_class = target_boxes_per_class
        self.rng = np.random.default_rng(seed=42)

        self.train_records: dict[str, tuple[Path, Path, list[list[float]]]] = {}
        self.class_to_stems: dict[int, list[str]] = {class_id: [] for class_id in YOLO_CLASS_NAMES}
        self.class_box_counts: dict[int, int] = {class_id: 0 for class_id in YOLO_CLASS_NAMES}
        self.initial_class_counts: dict[int, int] = {}

    def _prepare_output(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        shutil.copytree(self.source_dir, self.output_dir)

    def _load_train_metadata(self) -> None:
        img_dir = self.output_dir / "images" / "train"
        lbl_dir = self.output_dir / "labels" / "train"

        for img_path in sorted(img_dir.glob("*.png")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            labels = load_yolo_labels(lbl_path)
            self.train_records[img_path.stem] = (img_path, lbl_path, labels)
            present_classes = set()
            for cls_id, *_ in labels:
                cls = int(cls_id)
                self.class_box_counts[cls] = self.class_box_counts.get(cls, 0) + 1
                present_classes.add(cls)
            for cls in present_classes:
                self.class_to_stems.setdefault(cls, []).append(img_path.stem)

        self.initial_class_counts = dict(self.class_box_counts)

    def _choose_target(self) -> int:
        if self.target_boxes_per_class is not None:
            return int(self.target_boxes_per_class)
        max_count = max(self.class_box_counts.values()) if self.class_box_counts else 0
        return max_count

    def _apply_aug(self, image: np.ndarray, labels: list[list[float]], mode: str) -> tuple[np.ndarray, list[list[float]]]:
        if mode == "hflip":
            return cv2.flip(image, 1), transform_labels_hflip(labels)
        if mode == "vflip":
            return cv2.flip(image, 0), transform_labels_vflip(labels)
        if mode == "rot90":
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), transform_labels_rot90(labels)
        if mode == "rot180":
            return cv2.rotate(image, cv2.ROTATE_180), transform_labels_rot180(labels)
        if mode == "bright":
            bright = np.clip(image.astype(np.float32) * 1.20, 0, 255).astype(np.uint8)
            return bright, labels
        if mode == "noise":
            noise = self.rng.normal(0.0, 10.0, image.shape).astype(np.float32)
            noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            return noisy, labels
        return image, labels

    def run(self) -> BalanceSummary:
        print("\n" + "#" * 70)
        print("# STEP 7: CLASS-BALANCED TILED DATASET")
        print("#" * 70)
        print(f"[INFO] Source dataset: {self.source_dir}")
        print(f"[INFO] Output dataset: {self.output_dir}")

        self._prepare_output()
        self._load_train_metadata()
        target = self._choose_target()
        print(f"[INFO] Target train boxes per class: {target}")

        impossible_classes: list[int] = []
        for class_id in sorted(YOLO_CLASS_NAMES):
            if self.class_box_counts.get(class_id, 0) < target and not self.class_to_stems.get(class_id):
                impossible_classes.append(class_id)

        modes = ("hflip", "vflip", "rot90", "rot180", "bright", "noise")
        next_id = 0
        max_new_images = 20000
        added_images = 0

        def deficits() -> dict[int, int]:
            return {
                class_id: max(0, target - self.class_box_counts.get(class_id, 0))
                for class_id in YOLO_CLASS_NAMES
            }

        current_deficits = deficits()
        while any(v > 0 for v in current_deficits.values()) and added_images < max_new_images:
            candidate_classes = [c for c, d in current_deficits.items() if d > 0 and self.class_to_stems.get(c)]
            if not candidate_classes:
                break

            focus_class = max(candidate_classes, key=lambda c: current_deficits[c])
            stem = self.rng.choice(self.class_to_stems[focus_class])
            img_path, _, labels = self.train_records[stem]
            image = cv2.imread(str(img_path))
            if image is None or not labels:
                current_deficits[focus_class] = 0
                continue

            mode = modes[next_id % len(modes)]
            aug_image, aug_labels = self._apply_aug(image, labels, mode)
            new_stem = f"{stem}_cb_{next_id:06d}"
            out_img = self.output_dir / "images" / "train" / f"{new_stem}.png"
            out_lbl = self.output_dir / "labels" / "train" / f"{new_stem}.txt"

            cv2.imwrite(str(out_img), aug_image)
            save_yolo_labels(aug_labels, out_lbl)

            present_classes = set()
            for cls_id, *_ in aug_labels:
                cls = int(cls_id)
                self.class_box_counts[cls] = self.class_box_counts.get(cls, 0) + 1
                present_classes.add(cls)
            for cls in present_classes:
                self.class_to_stems.setdefault(cls, []).append(new_stem)

            self.train_records[new_stem] = (out_img, out_lbl, aug_labels)
            added_images += 1
            next_id += 1
            current_deficits = deficits()

            if added_images % 250 == 0:
                remaining = sum(current_deficits.values())
                print(f"[INFO] Added {added_images} images, remaining box deficit: {remaining}")

        ensure_dataset_yaml(self.output_dir)

        summary = BalanceSummary(
            source_dataset=str(self.source_dir),
            target_boxes_per_class=target,
            initial_train_images=len([p for p in (self.source_dir / "images" / "train").glob("*.png")]),
            final_train_images=len([p for p in (self.output_dir / "images" / "train").glob("*.png")]),
            added_train_images=added_images,
            initial_class_box_counts=self.initial_class_counts,
            final_class_box_counts=dict(self.class_box_counts),
            impossible_classes=impossible_classes,
        )

        summary_path = self.output_dir / "class_balance_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("[SUCCESS] Class-balancing expansion completed.")
        print(f"[SAVED] {summary_path}")
        print(f"[INFO] Added train images: {added_images}")
        if impossible_classes:
            print(f"[WARNING] Classes without train positives (cannot be balanced): {impossible_classes}")
        print("=" * 70)
        return summary


def choose_source_dataset() -> Path:
    if dataset_has_required_layout(EXPANDED_TILED_DATASET_DIR):
        return EXPANDED_TILED_DATASET_DIR
    return TILED_DATASET_DIR


def main() -> int:
    source = choose_source_dataset()
    balancer = TiledDatasetBalancer(source, BALANCED_TILED_DATASET_DIR)
    balancer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

