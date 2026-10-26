from __future__ import annotations

"""
Refine PF-MDD tiled dataset:
1) Rebuild train/val/test split for more stable validation/testing signals.
2) Boost train split so each class has at least N boxes (default: 50).
"""

import json
import random
import shutil
import sys
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from project_config import TILED_DATASET_DIR, YOLO_CLASS_NAMES, ensure_dataset_yaml

sys.stdout.reconfigure(encoding="utf-8")


SPLITS = ("train", "val", "test")
AUG_MODES = ("hflip", "vflip", "rot90", "rot180", "bright", "noise")


@dataclass
class Record:
    stem: str
    image_path: Path
    label_path: Path | None
    boxes: list[list[float]]
    class_counts: Counter
    group_key: str
    is_augmented: bool
    source_split: str

    @property
    def is_positive(self) -> bool:
        return bool(self.boxes)


def load_labels(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    labels: list[list[float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        labels.append([float(cls_id), xc, yc, bw, bh])
    return labels


def save_labels(labels: list[list[float]], label_path: Path) -> None:
    if not labels:
        return
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as handle:
        for cls_id, xc, yc, bw, bh in labels:
            handle.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def is_augmented_stem(stem: str) -> bool:
    markers = ("_hflip", "_vflip", "_rot90", "_rot180", "_bright", "_noise", "_cb_", "_boost_c")
    return any(marker in stem for marker in markers)


def origin_group_key(stem: str) -> str:
    # Base sample id format: 0001_002_00[_...]
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
        return "_".join(parts[:3])
    return stem


def aggregate_class_counts(records: list[Record]) -> Counter:
    out: Counter = Counter()
    for record in records:
        out.update(record.class_counts)
    return out


def build_records(dataset_dir: Path) -> list[Record]:
    records: list[Record] = []
    seen_stems = set()

    for split in SPLITS:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        for image_path in sorted(image_dir.glob("*.png")):
            stem = image_path.stem
            if stem in seen_stems:
                # Keep first occurrence only to avoid accidental duplicates across splits.
                continue
            seen_stems.add(stem)

            label_path = label_dir / f"{stem}.txt"
            boxes = load_labels(label_path)
            counts = Counter(int(box[0]) for box in boxes)
            records.append(
                Record(
                    stem=stem,
                    image_path=image_path,
                    label_path=label_path if label_path.exists() else None,
                    boxes=boxes,
                    class_counts=counts,
                    group_key=origin_group_key(stem),
                    is_augmented=is_augmented_stem(stem),
                    source_split=split,
                )
            )
    return records


def eval_target_per_split(total_boxes: int) -> int:
    if total_boxes >= 120:
        return 20
    if total_boxes >= 60:
        return 12
    if total_boxes >= 30:
        return 6
    if total_boxes >= 10:
        return 2
    if total_boxes >= 3:
        return 1
    return 0


def compute_class_targets(total_counts: Counter, min_eval_boxes_per_class: int) -> dict[str, dict[int, int]]:
    targets: dict[str, dict[int, int]] = {split: {} for split in SPLITS}
    for class_id in sorted(YOLO_CLASS_NAMES):
        total = int(total_counts.get(class_id, 0))
        # Reserve at least one box for train where possible.
        max_each_eval = max(0, (total - 1) // 2)
        eval_target = min(int(min_eval_boxes_per_class), max_each_eval)
        val_target = eval_target
        test_target = eval_target

        # Keep at least one training sample for any class that exists.
        while total > 0 and (val_target + test_target) >= total:
            if test_target >= val_target and test_target > 0:
                test_target -= 1
            elif val_target > 0:
                val_target -= 1
            else:
                break

        train_target = max(0, total - val_target - test_target)
        targets["train"][class_id] = train_target
        targets["val"][class_id] = val_target
        targets["test"][class_id] = test_target
    return targets


def choose_split_for_group(
    group_records: list[Record],
    current_counts: dict[str, Counter],
    class_targets: dict[str, dict[int, int]],
    current_pos_images: Counter,
    target_pos_images: dict[str, int],
) -> str:
    group_counts = aggregate_class_counts(group_records)
    group_size = len(group_records)

    best_split = "train"
    best_score = -10**9

    for split in SPLITS:
        gain = 0.0
        for class_id, amount in group_counts.items():
            need = max(0, class_targets[split].get(class_id, 0) - current_counts[split].get(class_id, 0))
            gain += min(need, amount)

        # Keep split sizes close to expected positive-image ratio.
        over = max(0, (current_pos_images[split] + group_size) - target_pos_images[split])
        penalty = over * 0.2
        score = gain - penalty

        if score > best_score:
            best_score = score
            best_split = split
    return best_split


def apply_aug(image: np.ndarray, labels: list[list[float]], mode: str, rng: np.random.Generator) -> tuple[np.ndarray, list[list[float]]]:
    if mode == "hflip":
        aug = cv2.flip(image, 1)
        out = [[c, 1.0 - xc, yc, bw, bh] for c, xc, yc, bw, bh in labels]
        return aug, out
    if mode == "vflip":
        aug = cv2.flip(image, 0)
        out = [[c, xc, 1.0 - yc, bw, bh] for c, xc, yc, bw, bh in labels]
        return aug, out
    if mode == "rot90":
        aug = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        out = [[c, 1.0 - yc, xc, bh, bw] for c, xc, yc, bw, bh in labels]
        return aug, out
    if mode == "rot180":
        aug = cv2.rotate(image, cv2.ROTATE_180)
        out = [[c, 1.0 - xc, 1.0 - yc, bw, bh] for c, xc, yc, bw, bh in labels]
        return aug, out
    if mode == "bright":
        aug = np.clip(image.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        return aug, labels
    if mode == "noise":
        noise = rng.normal(0.0, 10.0, image.shape).astype(np.float32)
        aug = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return aug, labels
    return image, labels


class DatasetRefiner:
    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        min_train_boxes_per_class: int = 50,
        min_eval_boxes_per_class: int = 20,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.min_train_boxes_per_class = min_train_boxes_per_class
        self.min_eval_boxes_per_class = min_eval_boxes_per_class
        self.split_ratios = split_ratios
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.summary: dict = {}

    def _prepare_output(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        for split in SPLITS:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _copy_record_to_split(self, record: Record, split: str) -> None:
        out_img = self.output_dir / "images" / split / f"{record.stem}.png"
        shutil.copy2(record.image_path, out_img)
        if record.boxes:
            out_lbl = self.output_dir / "labels" / split / f"{record.stem}.txt"
            save_labels(record.boxes, out_lbl)

    def _collect_output_train_records(self) -> list[Record]:
        train_images = self.output_dir / "images" / "train"
        train_labels = self.output_dir / "labels" / "train"
        records: list[Record] = []
        for image_path in sorted(train_images.glob("*.png")):
            stem = image_path.stem
            label_path = train_labels / f"{stem}.txt"
            boxes = load_labels(label_path)
            records.append(
                Record(
                    stem=stem,
                    image_path=image_path,
                    label_path=label_path if label_path.exists() else None,
                    boxes=boxes,
                    class_counts=Counter(int(box[0]) for box in boxes),
                    group_key=origin_group_key(stem),
                    is_augmented=is_augmented_stem(stem),
                    source_split="train",
                )
            )
        return records

    def _train_counts(self) -> Counter:
        counts = Counter()
        for label_path in (self.output_dir / "labels" / "train").glob("*.txt"):
            for box in load_labels(label_path):
                counts[int(box[0])] += 1
        return counts

    def _split_counts(self) -> dict[str, Counter]:
        out = {split: Counter() for split in SPLITS}
        for split in SPLITS:
            for label_path in (self.output_dir / "labels" / split).glob("*.txt"):
                for box in load_labels(label_path):
                    out[split][int(box[0])] += 1
        return out

    def _ensure_min_train_class_coverage(
        self,
        groups: dict[str, list[Record]],
        split_assignment: dict[str, str],
    ) -> dict:
        """
        Ensure every class present in the dataset has at least one train box,
        while preserving origin-level split isolation (moves happen by whole group).
        """
        split_counts = {split: Counter() for split in SPLITS}
        total_counts = Counter()
        for group_key, group_records in groups.items():
            split = split_assignment.get(group_key, "train")
            group_counter = aggregate_class_counts([r for r in group_records if not r.is_augmented])
            split_counts[split].update(group_counter)
            total_counts.update(group_counter)

        moved_groups: list[str] = []
        moved_box_counts = Counter()
        for class_id in sorted(YOLO_CLASS_NAMES):
            if total_counts.get(class_id, 0) <= 0:
                continue
            if split_counts["train"].get(class_id, 0) > 0:
                continue

            candidates: list[tuple[int, str, str, Counter]] = []
            for group_key, group_records in groups.items():
                from_split = split_assignment.get(group_key, "train")
                if from_split == "train":
                    continue
                group_counter = aggregate_class_counts([r for r in group_records if not r.is_augmented])
                if group_counter.get(class_id, 0) <= 0:
                    continue
                # Prefer moving a small group to minimize side effects on eval sets.
                group_size = len([r for r in group_records if not r.is_augmented])
                candidates.append((group_size, group_key, from_split, group_counter))

            if not candidates:
                continue

            candidates.sort(key=lambda item: item[0])
            _, selected_group, from_split, selected_counter = candidates[0]
            split_assignment[selected_group] = "train"
            split_counts[from_split].subtract(selected_counter)
            split_counts["train"].update(selected_counter)
            moved_box_counts.update(selected_counter)
            moved_groups.append(selected_group)

        return {
            "moved_groups_to_train": len(moved_groups),
            "moved_group_keys_examples": moved_groups[:12],
            "moved_box_counts_to_train": {str(k): int(v) for k, v in sorted(moved_box_counts.items())},
        }

    def _ensure_eval_class_coverage(
        self,
        groups: dict[str, list[Record]],
        split_assignment: dict[str, str],
    ) -> dict:
        """
        Improve val/test class presence by moving whole origin groups (no leakage).
        Target: for classes with total boxes >= 2, try to keep >=1 box in val and test.
        """
        split_counts = {split: Counter() for split in SPLITS}
        total_counts = Counter()
        group_counters: dict[str, Counter] = {}
        group_sizes: dict[str, int] = {}

        for group_key, group_records in groups.items():
            group_counter = aggregate_class_counts(group_records)
            group_counters[group_key] = group_counter
            group_sizes[group_key] = len(group_records)
            split = split_assignment.get(group_key, "train")
            split_counts[split].update(group_counter)
            total_counts.update(group_counter)

        moved_to_val = Counter()
        moved_to_test = Counter()
        moved_groups_to_val: list[str] = []
        moved_groups_to_test: list[str] = []

        def pick_group_for_class(class_id: int, target_split: str) -> tuple[str, str] | None:
            candidates: list[tuple[int, int, str, str]] = []
            for group_key, group_counter in group_counters.items():
                if group_counter.get(class_id, 0) <= 0:
                    continue
                from_split = split_assignment.get(group_key, "train")
                if from_split == target_split:
                    continue

                # Keep at least one box for this class in source split.
                remaining = split_counts[from_split].get(class_id, 0) - group_counter.get(class_id, 0)
                if remaining <= 0:
                    continue

                # Prefer moving from train first, then smaller groups.
                split_penalty = 0 if from_split == "train" else 1
                candidates.append((split_penalty, group_sizes[group_key], group_key, from_split))

            if not candidates:
                return None
            candidates.sort()
            _, _, gk, fs = candidates[0]
            return gk, fs

        for class_id in sorted(YOLO_CLASS_NAMES):
            total = int(total_counts.get(class_id, 0))
            if total < 2:
                continue

            while split_counts["val"].get(class_id, 0) <= 0:
                picked = pick_group_for_class(class_id, "val")
                if picked is None:
                    break
                gk, fs = picked
                gc = group_counters[gk]
                split_assignment[gk] = "val"
                split_counts[fs].subtract(gc)
                split_counts["val"].update(gc)
                moved_to_val.update(gc)
                moved_groups_to_val.append(gk)

            while split_counts["test"].get(class_id, 0) <= 0:
                picked = pick_group_for_class(class_id, "test")
                if picked is None:
                    break
                gk, fs = picked
                gc = group_counters[gk]
                split_assignment[gk] = "test"
                split_counts[fs].subtract(gc)
                split_counts["test"].update(gc)
                moved_to_test.update(gc)
                moved_groups_to_test.append(gk)

        return {
            "moved_groups_to_val": len(moved_groups_to_val),
            "moved_groups_to_test": len(moved_groups_to_test),
            "moved_group_keys_to_val_examples": moved_groups_to_val[:12],
            "moved_group_keys_to_test_examples": moved_groups_to_test[:12],
            "moved_box_counts_to_val": {str(k): int(v) for k, v in sorted(moved_to_val.items())},
            "moved_box_counts_to_test": {str(k): int(v) for k, v in sorted(moved_to_test.items())},
        }

    def _boost_train_minority(self) -> dict:
        train_records = self._collect_output_train_records()
        train_by_class: dict[int, list[Record]] = defaultdict(list)
        for record in train_records:
            for cls in record.class_counts:
                train_by_class[cls].append(record)

        added = 0
        impossible: list[int] = []
        boosts_per_class: dict[int, int] = {}
        class_counts = self._train_counts()

        for class_id in sorted(YOLO_CLASS_NAMES):
            target = self.min_train_boxes_per_class
            current = int(class_counts.get(class_id, 0))
            if current >= target:
                continue

            base_records = train_by_class.get(class_id, [])

            if not base_records:
                impossible.append(class_id)
                boosts_per_class[class_id] = 0
                continue

            idx = 0
            class_added = 0
            while current < target:
                base = base_records[idx % len(base_records)]
                mode = AUG_MODES[idx % len(AUG_MODES)]
                idx += 1

                image = cv2.imread(str(base.image_path))
                if image is None:
                    continue
                labels = load_labels(base.label_path) if base.label_path else []
                if not labels:
                    continue

                aug_image, aug_labels = apply_aug(image, labels, mode, self.np_rng)
                new_stem = f"{base.stem}_boost_c{class_id}_{idx:06d}"
                out_img = self.output_dir / "images" / "train" / f"{new_stem}.png"
                out_lbl = self.output_dir / "labels" / "train" / f"{new_stem}.txt"
                cv2.imwrite(str(out_img), aug_image)
                save_labels(aug_labels, out_lbl)

                # Update counts using all labels in this generated image.
                generated_counts = Counter(int(b[0]) for b in aug_labels)
                class_counts.update(generated_counts)
                current = int(class_counts.get(class_id, 0))
                added += 1
                class_added += 1

            boosts_per_class[class_id] = class_added

        return {
            "added_train_images": added,
            "impossible_classes": impossible,
            "boosts_per_class": {str(k): v for k, v in boosts_per_class.items()},
            "final_train_box_counts": {str(k): int(class_counts.get(k, 0)) for k in sorted(YOLO_CLASS_NAMES)},
        }

    def _remove_train_hashes_present_in_eval(self) -> dict:
        """
        Remove exact duplicate train images whose pixel hash also appears in val/test.
        This avoids cross-split exact-image leakage.
        """
        def file_sha256(path: Path) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        eval_hashes: set[str] = set()
        for split in ("val", "test"):
            for image_path in (self.output_dir / "images" / split).glob("*.png"):
                eval_hashes.add(file_sha256(image_path))

        removed_images = 0
        removed_labels = 0
        removed_stems: list[str] = []
        for image_path in sorted((self.output_dir / "images" / "train").glob("*.png")):
            if file_sha256(image_path) not in eval_hashes:
                continue
            stem = image_path.stem
            label_path = self.output_dir / "labels" / "train" / f"{stem}.txt"
            image_path.unlink(missing_ok=True)
            if label_path.exists():
                label_path.unlink()
                removed_labels += 1
            removed_images += 1
            if len(removed_stems) < 12:
                removed_stems.append(stem)

        return {
            "removed_train_images": removed_images,
            "removed_train_labels": removed_labels,
            "removed_train_examples": removed_stems,
        }

    def _remove_test_hashes_present_in_val(self) -> dict:
        """
        Remove exact duplicate test images that also appear in validation.
        Keep validation unchanged and make test independent from val.
        """
        def file_sha256(path: Path) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        val_hashes: set[str] = set()
        for image_path in (self.output_dir / "images" / "val").glob("*.png"):
            val_hashes.add(file_sha256(image_path))

        removed_images = 0
        removed_labels = 0
        removed_stems: list[str] = []
        for image_path in sorted((self.output_dir / "images" / "test").glob("*.png")):
            if file_sha256(image_path) not in val_hashes:
                continue
            stem = image_path.stem
            label_path = self.output_dir / "labels" / "test" / f"{stem}.txt"
            image_path.unlink(missing_ok=True)
            if label_path.exists():
                label_path.unlink()
                removed_labels += 1
            removed_images += 1
            if len(removed_stems) < 12:
                removed_stems.append(stem)

        return {
            "removed_test_images": removed_images,
            "removed_test_labels": removed_labels,
            "removed_test_examples": removed_stems,
        }

    def run(self) -> dict:
        print("\n" + "#" * 72)
        print("# STEP 8: REFINE SPLIT + BOOST MINORITY CLASSES")
        print("#" * 72)
        print(f"[INFO] Source dataset: {self.source_dir}")
        print(f"[INFO] Output dataset: {self.output_dir}")
        print(f"[INFO] Min train boxes/class target: {self.min_train_boxes_per_class}")
        print(f"[INFO] Min eval boxes/class target: {self.min_eval_boxes_per_class}")

        records = build_records(self.source_dir)
        groups: dict[str, list[Record]] = defaultdict(list)
        base_groups: dict[str, list[Record]] = defaultdict(list)
        for r in records:
            groups[r.group_key].append(r)
            if not r.is_augmented:
                base_groups[r.group_key].append(r)

        positive_groups = [(k, v) for k, v in base_groups.items() if any(r.is_positive for r in v)]
        background_groups = [(k, v) for k, v in base_groups.items() if not any(r.is_positive for r in v)]
        self.rng.shuffle(positive_groups)
        self.rng.shuffle(background_groups)

        total_positive_boxes = Counter()
        for _, group_records in positive_groups:
            total_positive_boxes.update(aggregate_class_counts(group_records))
        class_targets = compute_class_targets(total_positive_boxes, self.min_eval_boxes_per_class)

        # Positive-group split (origin-level; no leakage across splits).
        current_counts = {split: Counter() for split in SPLITS}
        split_assignment: dict[str, str] = {}
        current_pos_images: Counter = Counter()

        total_pos_images = sum(len(g) for _, g in positive_groups)
        train_r, val_r, test_r = self.split_ratios
        target_pos_images = {
            "train": int(round(total_pos_images * train_r)),
            "val": int(round(total_pos_images * val_r)),
            "test": total_pos_images - int(round(total_pos_images * train_r)) - int(round(total_pos_images * val_r)),
        }

        positive_groups_sorted = sorted(
            positive_groups,
            key=lambda kv: sum(aggregate_class_counts(kv[1]).values()),
            reverse=True,
        )
        for group_key, group_records in positive_groups_sorted:
            split = choose_split_for_group(
                group_records=group_records,
                current_counts=current_counts,
                class_targets=class_targets,
                current_pos_images=current_pos_images,
                target_pos_images=target_pos_images,
            )
            split_assignment[group_key] = split
            current_counts[split].update(aggregate_class_counts(group_records))
            current_pos_images[split] += len(group_records)

        # Background-group split by ratio.
        total_bg_images = sum(len(g) for _, g in background_groups)
        target_bg_images = {
            "train": int(round(total_bg_images * train_r)),
            "val": int(round(total_bg_images * val_r)),
            "test": total_bg_images - int(round(total_bg_images * train_r)) - int(round(total_bg_images * val_r)),
        }
        current_bg_images = Counter()
        for group_key, group_records in background_groups:
            deficits = {split: target_bg_images[split] - current_bg_images[split] for split in SPLITS}
            split = max(SPLITS, key=lambda s: deficits[s])
            split_assignment[group_key] = split
            current_bg_images[split] += len(group_records)

        split_inconsistencies: dict[str, list[str]] = {}

        # Fallback for groups that only contain augmented samples.
        for group_key in groups.keys():
            split_assignment.setdefault(group_key, "train")

        eval_coverage_info = self._ensure_eval_class_coverage(groups=base_groups, split_assignment=split_assignment)
        train_coverage_info = self._ensure_min_train_class_coverage(groups=base_groups, split_assignment=split_assignment)

        # Build output dataset.
        self._prepare_output()
        for group_key, group_records in groups.items():
            split = split_assignment[group_key]
            for record in group_records:
                # Keep validation/testing "clean": no augmented samples outside train.
                if split != "train" and record.is_augmented:
                    continue
                target_split = "train" if record.is_augmented else split
                self._copy_record_to_split(record, target_split)

        ensure_dataset_yaml(self.output_dir)
        pre_boost_counts = self._split_counts()
        boost_info = self._boost_train_minority()
        duplicate_cleanup_info = self._remove_train_hashes_present_in_eval()
        eval_split_cleanup_info = self._remove_test_hashes_present_in_val()
        post_boost_counts = self._split_counts()
        yaml_path = ensure_dataset_yaml(self.output_dir)

        def split_augmented_counts() -> dict[str, int]:
            out: dict[str, int] = {}
            for split in SPLITS:
                images = list((self.output_dir / "images" / split).glob("*.png"))
                out[split] = sum(1 for path in images if is_augmented_stem(path.stem))
            return out

        def split_image_counts() -> dict[str, int]:
            return {
                split: len(list((self.output_dir / "images" / split).glob("*.png")))
                for split in SPLITS
            }

        def split_label_counts() -> dict[str, int]:
            return {
                split: len(list((self.output_dir / "labels" / split).glob("*.txt")))
                for split in SPLITS
            }

        self.summary = {
            "source_dataset": str(self.source_dir),
            "output_dataset": str(self.output_dir),
            "min_train_boxes_per_class": self.min_train_boxes_per_class,
            "image_counts": split_image_counts(),
            "augmented_image_counts": split_augmented_counts(),
            "label_counts": split_label_counts(),
            "pre_boost_box_counts": {
                split: {str(k): int(v) for k, v in sorted(pre_boost_counts[split].items())}
                for split in SPLITS
            },
            "post_boost_box_counts": {
                split: {str(k): int(v) for k, v in sorted(post_boost_counts[split].items())}
                for split in SPLITS
            },
            "class_targets_for_split": {
                split: {str(k): int(v) for k, v in sorted(class_targets[split].items())}
                for split in SPLITS
            },
            "train_coverage_adjustments": train_coverage_info,
            "eval_coverage_adjustments": eval_coverage_info,
            "source_split_inconsistencies": split_inconsistencies,
            "boost_info": boost_info,
            "duplicate_hash_cleanup": duplicate_cleanup_info,
            "eval_split_hash_cleanup": eval_split_cleanup_info,
            "dataset_yaml": str(yaml_path),
        }

        summary_path = self.output_dir / "refine_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(self.summary, handle, indent=2, ensure_ascii=False)

        print("\n" + "=" * 72)
        print("[SUCCESS] Refinement completed.")
        print(f"[SAVED] Dataset YAML: {yaml_path}")
        print(f"[SAVED] Summary: {summary_path}")
        print(f"[INFO] Image counts: {self.summary['image_counts']}")
        print(f"[INFO] Label counts: {self.summary['label_counts']}")
        print(f"[INFO] Boosted images: {boost_info['added_train_images']}")
        if boost_info["impossible_classes"]:
            print(f"[WARNING] Classes impossible to boost: {boost_info['impossible_classes']}")
        print("=" * 72)
        return self.summary


def main() -> int:
    source = TILED_DATASET_DIR
    output = source.parent / "pfmd_tiled_dataset_refined"
    refiner = DatasetRefiner(
        source_dir=source,
        output_dir=output,
        min_train_boxes_per_class=50,
        min_eval_boxes_per_class=20,
    )
    refiner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
