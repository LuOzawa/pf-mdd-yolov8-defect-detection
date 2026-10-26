from __future__ import annotations

"""
Expand the tiled PF-MDD dataset by augmenting positive training tiles.

Design:
- Keep val/test untouched for fair evaluation.
- Copy all original train images.
- Apply augmentations only to train samples that have non-empty YOLO labels.
- Save an expanded dataset to `pfmd_tiled_dataset_expanded`.
"""

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from project_config import (
    EXPANDED_TILED_DATASET_DIR,
    TILED_DATASET_DIR,
    ensure_dataset_yaml,
)

sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class ExpansionStats:
    original_train_images: int = 0
    original_train_positive_images: int = 0
    original_train_background_images: int = 0
    augmented_train_images: int = 0
    augmented_train_labels: int = 0
    copied_val_images: int = 0
    copied_test_images: int = 0

    def to_dict(self) -> dict:
        return {
            "original_train_images": self.original_train_images,
            "original_train_positive_images": self.original_train_positive_images,
            "original_train_background_images": self.original_train_background_images,
            "augmented_train_images": self.augmented_train_images,
            "augmented_train_labels": self.augmented_train_labels,
            "copied_val_images": self.copied_val_images,
            "copied_test_images": self.copied_test_images,
            "total_train_images_after_expansion": self.original_train_images + self.augmented_train_images,
            "augmentation_factor_on_positive": (
                (self.original_train_positive_images + self.augmented_train_labels) / self.original_train_positive_images
                if self.original_train_positive_images
                else 0.0
            ),
        }


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


def save_yolo_labels(labels: list[list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for cls_id, xc, yc, bw, bh in labels:
            handle.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def flip_horizontal(labels: list[list[float]]) -> list[list[float]]:
    out = []
    for cls_id, xc, yc, bw, bh in labels:
        out.append([cls_id, 1.0 - xc, yc, bw, bh])
    return out


def flip_vertical(labels: list[list[float]]) -> list[list[float]]:
    out = []
    for cls_id, xc, yc, bw, bh in labels:
        out.append([cls_id, xc, 1.0 - yc, bw, bh])
    return out


class TiledDatasetExpander:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.stats = ExpansionStats()
        self.rng = np.random.default_rng(seed=42)

    def _prepare_output(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        for split in ("train", "val", "test"):
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _copy_split_without_augmentation(self, split: str) -> None:
        image_in = self.input_dir / "images" / split
        label_in = self.input_dir / "labels" / split
        image_out = self.output_dir / "images" / split
        label_out = self.output_dir / "labels" / split

        copied = 0
        for img_path in sorted(image_in.glob("*.png")):
            shutil.copy2(img_path, image_out / img_path.name)
            lbl_path = label_in / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, label_out / lbl_path.name)
            copied += 1

        if split == "val":
            self.stats.copied_val_images = copied
        elif split == "test":
            self.stats.copied_test_images = copied

    def _augment_train_split(self) -> None:
        image_in = self.input_dir / "images" / "train"
        label_in = self.input_dir / "labels" / "train"
        image_out = self.output_dir / "images" / "train"
        label_out = self.output_dir / "labels" / "train"

        aug_methods = ("hflip", "vflip", "bright", "noise")

        for img_path in sorted(image_in.glob("*.png")):
            # Always keep original train image.
            shutil.copy2(img_path, image_out / img_path.name)
            self.stats.original_train_images += 1

            lbl_path = label_in / f"{img_path.stem}.txt"
            labels = load_yolo_labels(lbl_path)

            if labels:
                shutil.copy2(lbl_path, label_out / lbl_path.name)
                self.stats.original_train_positive_images += 1
            else:
                self.stats.original_train_background_images += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            for aug in aug_methods:
                aug_name = f"{img_path.stem}_{aug}"
                aug_img_path = image_out / f"{aug_name}.png"
                aug_lbl_path = label_out / f"{aug_name}.txt"

                if aug == "hflip":
                    aug_img = cv2.flip(image, 1)
                    aug_labels = flip_horizontal(labels)
                elif aug == "vflip":
                    aug_img = cv2.flip(image, 0)
                    aug_labels = flip_vertical(labels)
                elif aug == "bright":
                    aug_img = np.clip(image.astype(np.float32) * 1.20, 0, 255).astype(np.uint8)
                    aug_labels = labels
                elif aug == "noise":
                    noise = self.rng.normal(0.0, 10.0, image.shape).astype(np.float32)
                    aug_img = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                    aug_labels = labels
                else:
                    continue

                cv2.imwrite(str(aug_img_path), aug_img)
                save_yolo_labels(aug_labels, aug_lbl_path)
                self.stats.augmented_train_images += 1
                self.stats.augmented_train_labels += 1

    def run(self) -> ExpansionStats:
        print("\n" + "#" * 70)
        print("# STEP 6: EXPAND TILED DATASET")
        print("#" * 70)
        print(f"[INFO] Input dataset: {self.input_dir}")
        print(f"[INFO] Output dataset: {self.output_dir}")

        self._prepare_output()
        self._augment_train_split()
        self._copy_split_without_augmentation("val")
        self._copy_split_without_augmentation("test")

        yaml_path = ensure_dataset_yaml(self.output_dir)
        summary_path = self.output_dir / "expansion_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(self.stats.to_dict(), handle, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("[SUCCESS] Dataset expansion completed.")
        print(f"[SAVED] Dataset YAML: {yaml_path}")
        print(f"[SAVED] Summary: {summary_path}")
        for key, value in self.stats.to_dict().items():
            print(f"  - {key}: {value}")
        print("=" * 70)
        return self.stats


def main() -> int:
    expander = TiledDatasetExpander(TILED_DATASET_DIR, EXPANDED_TILED_DATASET_DIR)
    expander.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
