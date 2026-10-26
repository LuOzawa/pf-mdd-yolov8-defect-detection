from __future__ import annotations

"""
Expand PF-MDD refined dataset by generating additional simulated samples for train split only.

Design goals:
- Keep val/test unchanged for fair evaluation.
- Increase train positive diversity and class box counts to a configurable target.
- Optionally add some background-only augmented tiles.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from project_config import REFINED_TILED_DATASET_DIR, YOLO_CLASS_NAMES, ensure_dataset_yaml

sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class TrainSample:
    stem: str
    image_path: Path
    label_path: Path | None
    labels: list[list[float]]

    @property
    def is_positive(self) -> bool:
        return bool(self.labels)


def load_labels(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    labels: list[list[float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        labels.append([float(cls_id), x, y, w, h])
    return labels


def save_labels(labels: list[list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for cls_id, x, y, w, h in labels:
            handle.write(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def is_simulated_stem(stem: str) -> bool:
    return "_sim_" in stem


def aug_hflip(image: np.ndarray, labels: list[list[float]]) -> tuple[np.ndarray, list[list[float]]]:
    out = [[c, 1.0 - x, y, w, h] for c, x, y, w, h in labels]
    return cv2.flip(image, 1), out


def aug_vflip(image: np.ndarray, labels: list[list[float]]) -> tuple[np.ndarray, list[list[float]]]:
    out = [[c, x, 1.0 - y, w, h] for c, x, y, w, h in labels]
    return cv2.flip(image, 0), out


def aug_rot90(image: np.ndarray, labels: list[list[float]]) -> tuple[np.ndarray, list[list[float]]]:
    out = [[c, 1.0 - y, x, h, w] for c, x, y, w, h in labels]
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), out


def aug_rot180(image: np.ndarray, labels: list[list[float]]) -> tuple[np.ndarray, list[list[float]]]:
    out = [[c, 1.0 - x, 1.0 - y, w, h] for c, x, y, w, h in labels]
    return cv2.rotate(image, cv2.ROTATE_180), out


def aug_brightness(image: np.ndarray, labels: list[list[float]], factor: float) -> tuple[np.ndarray, list[list[float]]]:
    out = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out, labels


def aug_gamma(image: np.ndarray, labels: list[list[float]], gamma: float) -> tuple[np.ndarray, list[list[float]]]:
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    out = cv2.LUT(image, lut)
    return out, labels


def aug_noise(image: np.ndarray, labels: list[list[float]], sigma: float, rng: np.random.Generator) -> tuple[np.ndarray, list[list[float]]]:
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    out = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out, labels


def aug_blur(image: np.ndarray, labels: list[list[float]], ksize: int) -> tuple[np.ndarray, list[list[float]]]:
    out = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return out, labels


class RefinedTrainSimulator:
    def __init__(
        self,
        dataset_dir: Path,
        target_boxes_per_class: int = 320,
        max_new_images: int = 6000,
        background_ratio: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.target_boxes_per_class = int(target_boxes_per_class)
        self.max_new_images = int(max_new_images)
        self.background_ratio = float(background_ratio)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.train_img_dir = self.dataset_dir / "images" / "train"
        self.train_lbl_dir = self.dataset_dir / "labels" / "train"

        self.samples: dict[str, TrainSample] = {}
        self.class_counts: Counter = Counter()
        self.class_to_stems: dict[int, list[str]] = defaultdict(list)
        self.base_positive_stems: dict[int, list[str]] = defaultdict(list)
        self.base_background_stems: list[str] = []

    def _load_train(self) -> None:
        for image_path in sorted(self.train_img_dir.glob("*.png")):
            stem = image_path.stem
            label_path = self.train_lbl_dir / f"{stem}.txt"
            labels = load_labels(label_path)
            sample = TrainSample(
                stem=stem,
                image_path=image_path,
                label_path=label_path if label_path.exists() else None,
                labels=labels,
            )
            self.samples[stem] = sample

            if labels:
                present = set()
                for cls_id, *_ in labels:
                    cls = int(cls_id)
                    self.class_counts[cls] += 1
                    present.add(cls)
                for cls in present:
                    self.class_to_stems[cls].append(stem)
                    if not is_simulated_stem(stem):
                        self.base_positive_stems[cls].append(stem)
            else:
                if not is_simulated_stem(stem):
                    self.base_background_stems.append(stem)

    def _unique_stem(self, base_stem: str, class_id: int, idx: int, mode: str) -> str:
        return f"{base_stem}_sim_c{class_id}_{mode}_{idx:06d}"

    def _apply_aug(self, image: np.ndarray, labels: list[list[float]], mode: str) -> tuple[np.ndarray, list[list[float]]]:
        if mode == "hflip":
            return aug_hflip(image, labels)
        if mode == "vflip":
            return aug_vflip(image, labels)
        if mode == "rot90":
            return aug_rot90(image, labels)
        if mode == "rot180":
            return aug_rot180(image, labels)
        if mode == "bright_up":
            return aug_brightness(image, labels, 1.25)
        if mode == "bright_down":
            return aug_brightness(image, labels, 0.82)
        if mode == "gamma_low":
            return aug_gamma(image, labels, 0.75)
        if mode == "gamma_high":
            return aug_gamma(image, labels, 1.35)
        if mode == "noise_8":
            return aug_noise(image, labels, 8.0, self.np_rng)
        if mode == "noise_14":
            return aug_noise(image, labels, 14.0, self.np_rng)
        if mode == "blur3":
            return aug_blur(image, labels, 3)
        if mode == "blur5":
            return aug_blur(image, labels, 5)
        return image, labels

    def _all_train_box_counts(self) -> dict[int, int]:
        counts = {int(c): 0 for c in YOLO_CLASS_NAMES}
        for label_path in self.train_lbl_dir.glob("*.txt"):
            for cls_id, *_ in load_labels(label_path):
                counts[int(cls_id)] += 1
        return counts

    def _add_positive_simulations(self) -> dict:
        modes = (
            "hflip",
            "vflip",
            "rot90",
            "rot180",
            "bright_up",
            "bright_down",
            "gamma_low",
            "gamma_high",
            "noise_8",
            "noise_14",
            "blur3",
            "blur5",
        )
        mode_idx = 0
        next_id = 0
        added = 0
        added_per_class: Counter = Counter()
        impossible: list[int] = []

        def deficits() -> dict[int, int]:
            return {
                class_id: max(0, self.target_boxes_per_class - int(self.class_counts.get(class_id, 0)))
                for class_id in YOLO_CLASS_NAMES
            }

        cur_def = deficits()
        while any(v > 0 for v in cur_def.values()) and added < self.max_new_images:
            candidates = [c for c, d in cur_def.items() if d > 0]
            if not candidates:
                break
            focus = max(candidates, key=lambda c: cur_def[c])
            base_pool = self.base_positive_stems.get(focus, [])
            if not base_pool:
                impossible.append(focus)
                cur_def[focus] = 0
                continue

            base_stem = self.rng.choice(base_pool)
            base_sample = self.samples[base_stem]
            image = cv2.imread(str(base_sample.image_path))
            if image is None or not base_sample.labels:
                cur_def[focus] = 0
                continue

            mode = modes[mode_idx % len(modes)]
            mode_idx += 1
            aug_img, aug_lbl = self._apply_aug(image, base_sample.labels, mode)

            new_stem = self._unique_stem(base_stem, focus, next_id, mode)
            next_id += 1
            out_img = self.train_img_dir / f"{new_stem}.png"
            out_lbl = self.train_lbl_dir / f"{new_stem}.txt"

            cv2.imwrite(str(out_img), aug_img)
            save_labels(aug_lbl, out_lbl)

            sample = TrainSample(
                stem=new_stem,
                image_path=out_img,
                label_path=out_lbl,
                labels=aug_lbl,
            )
            self.samples[new_stem] = sample

            present = set()
            for cls_id, *_ in aug_lbl:
                cls = int(cls_id)
                self.class_counts[cls] += 1
                present.add(cls)
            for cls in present:
                self.class_to_stems[cls].append(new_stem)
            added += 1
            added_per_class[focus] += 1
            cur_def = deficits()

        return {
            "added_positive_images": int(added),
            "added_images_per_focus_class": {str(k): int(v) for k, v in sorted(added_per_class.items())},
            "impossible_classes": sorted(set(int(c) for c in impossible)),
        }

    def _add_background_simulations(self, positive_added: int) -> dict:
        if positive_added <= 0 or not self.base_background_stems:
            return {
                "added_background_images": 0,
                "target_background_additions": 0,
            }

        target_bg = int(round(positive_added * self.background_ratio))
        modes = ("hflip", "vflip", "rot90", "rot180", "bright_up", "bright_down", "gamma_low", "gamma_high", "noise_8", "blur3")
        added = 0
        mode_idx = 0
        next_id = 0
        while added < target_bg and added < self.max_new_images:
            base_stem = self.rng.choice(self.base_background_stems)
            base_sample = self.samples[base_stem]
            image = cv2.imread(str(base_sample.image_path))
            if image is None:
                continue
            mode = modes[mode_idx % len(modes)]
            mode_idx += 1
            aug_img, _ = self._apply_aug(image, [], mode)
            new_stem = f"{base_stem}_sim_bg_{mode}_{next_id:06d}"
            next_id += 1
            out_img = self.train_img_dir / f"{new_stem}.png"
            cv2.imwrite(str(out_img), aug_img)
            self.samples[new_stem] = TrainSample(
                stem=new_stem,
                image_path=out_img,
                label_path=None,
                labels=[],
            )
            added += 1

        return {
            "added_background_images": int(added),
            "target_background_additions": int(target_bg),
        }

    def run(self) -> dict:
        if not self.train_img_dir.exists() or not self.train_lbl_dir.exists():
            raise FileNotFoundError(f"Invalid dataset layout: {self.dataset_dir}")

        print("\n" + "#" * 72)
        print("# STEP 10: EXPAND REFINED TRAIN WITH SIMULATED DATA")
        print("#" * 72)
        print(f"[INFO] Dataset: {self.dataset_dir}")
        print(f"[INFO] Target train boxes/class: {self.target_boxes_per_class}")
        print(f"[INFO] Max new images: {self.max_new_images}")
        print(f"[INFO] Background ratio: {self.background_ratio}")

        self._load_train()
        before_counts = self._all_train_box_counts()
        before_images = len(list(self.train_img_dir.glob("*.png")))
        before_labels = len(list(self.train_lbl_dir.glob("*.txt")))

        pos_info = self._add_positive_simulations()
        bg_info = self._add_background_simulations(pos_info["added_positive_images"])

        after_counts = self._all_train_box_counts()
        after_images = len(list(self.train_img_dir.glob("*.png")))
        after_labels = len(list(self.train_lbl_dir.glob("*.txt")))
        ensure_dataset_yaml(self.dataset_dir)

        summary = {
            "dataset_dir": str(self.dataset_dir),
            "target_boxes_per_class": self.target_boxes_per_class,
            "max_new_images": self.max_new_images,
            "background_ratio": self.background_ratio,
            "before_train_images": before_images,
            "before_train_labels": before_labels,
            "after_train_images": after_images,
            "after_train_labels": after_labels,
            "before_train_box_counts": {str(k): int(v) for k, v in sorted(before_counts.items())},
            "after_train_box_counts": {str(k): int(v) for k, v in sorted(after_counts.items())},
            "positive_simulation": pos_info,
            "background_simulation": bg_info,
        }

        summary_path = self.dataset_dir / "step10_simulation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        print("\n" + "=" * 72)
        print("[SUCCESS] Train simulation expansion completed.")
        print(f"[SAVED] {summary_path}")
        print(f"[INFO] Train images: {before_images} -> {after_images}")
        print(f"[INFO] Train labels: {before_labels} -> {after_labels}")
        print("=" * 72)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand refined train split with simulated data.")
    parser.add_argument("--dataset-dir", type=Path, default=REFINED_TILED_DATASET_DIR)
    parser.add_argument("--target-boxes-per-class", type=int, default=320)
    parser.add_argument("--max-new-images", type=int, default=6000)
    parser.add_argument("--background-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    simulator = RefinedTrainSimulator(
        dataset_dir=args.dataset_dir,
        target_boxes_per_class=args.target_boxes_per_class,
        max_new_images=args.max_new_images,
        background_ratio=args.background_ratio,
        seed=args.seed,
    )
    simulator.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
