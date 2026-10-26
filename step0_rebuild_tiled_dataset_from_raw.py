from __future__ import annotations

"""
Rebuild `pfmd_tiled_dataset` from raw PF-MDD defect images and YOLO labels.

Pipeline:
1) Split original (non-tiled) images by origin into train/val/test.
2) Tile each wide image (4096x256) into 640x640 overlapping tiles.
3) Convert original YOLO boxes into per-tile YOLO boxes.
"""

import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from image_processing_strategy import WideImageProcessor
from project_config import DATA_DIR, TILED_DATASET_DIR, ensure_dataset_yaml

sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class RawSample:
    stem: str
    image_path: Path
    labels: list[list[float]]

    @property
    def classes(self) -> set[int]:
        return {int(b[0]) for b in self.labels}

    @property
    def primary_class(self) -> int:
        if not self.labels:
            return -1
        counts = Counter(int(b[0]) for b in self.labels)
        return max(counts.items(), key=lambda kv: kv[1])[0]


def load_yolo_labels(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    out: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        cls_id = int(float(p[0]))
        x, y, w, h = map(float, p[1:5])
        out.append([float(cls_id), x, y, w, h])
    return out


class TiledDatasetRebuilder:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        output_dir: Path = TILED_DATASET_DIR,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.split_ratios = split_ratios
        self.rng = random.Random(seed)
        self.processor = WideImageProcessor(
            original_size=(4096, 256),
            target_tile_size=640,
            overlap_ratio=0.25,
        )

        self.defect_dir = self.data_dir / "Defect_images" / "Defect_images"
        self.label_dir = self.data_dir / "yolo_labels"

    def _prepare_output(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        for split in ("train", "val", "test"):
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _collect_raw_samples(self) -> list[RawSample]:
        samples: list[RawSample] = []
        for image_path in sorted(self.defect_dir.glob("*.png")):
            stem = image_path.stem
            labels = load_yolo_labels(self.label_dir / f"{stem}.txt")
            samples.append(RawSample(stem=stem, image_path=image_path, labels=labels))
        return samples

    def _split_raw_samples(self, samples: list[RawSample]) -> dict[str, list[RawSample]]:
        train_r, val_r, _ = self.split_ratios
        by_primary: dict[int, list[RawSample]] = defaultdict(list)
        for s in samples:
            by_primary[s.primary_class].append(s)

        splits = {"train": [], "val": [], "test": []}
        for cls, items in by_primary.items():
            self.rng.shuffle(items)
            n = len(items)
            n_train = int(round(n * train_r))
            n_val = int(round(n * val_r))

            # Keep at least one sample for train if class exists.
            if n > 0 and n_train <= 0:
                n_train = 1
            if n_train + n_val > n:
                n_val = max(0, n - n_train)
            n_test = n - n_train - n_val

            # For classes with enough samples, try to keep val/test both non-empty.
            if n >= 3:
                if n_val == 0 and n_train > 1:
                    n_val = 1
                    n_train -= 1
                if n_test == 0 and n_train > 1:
                    n_test = 1
                    n_train -= 1
            if n_train + n_val + n_test != n:
                n_test = n - n_train - n_val

            splits["train"].extend(items[:n_train])
            splits["val"].extend(items[n_train : n_train + n_val])
            splits["test"].extend(items[n_train + n_val :])

        return splits

    def _tile_and_save(self, split: str, samples: list[RawSample]) -> dict[str, int]:
        img_out = self.output_dir / "images" / split
        lbl_out = self.output_dir / "labels" / split
        tile_images = 0
        tile_labels = 0
        tile_boxes = 0

        for sample in samples:
            image = cv2.imread(str(sample.image_path))
            if image is None:
                continue
            tiles = self.processor.tile_image_with_overlap(image, return_metadata=True)
            for idx, (tile_img, tile_meta) in enumerate(tiles):
                tile_stem = f"{sample.stem}_t{idx:02d}"
                tile_img_path = img_out / f"{tile_stem}.png"
                cv2.imwrite(str(tile_img_path), tile_img)
                tile_images += 1

                if sample.labels and tile_meta is not None:
                    tile_boxes_list = self.processor.convert_yolo_boxes_for_tile(sample.labels, tile_meta)
                    if tile_boxes_list:
                        tile_lbl_path = lbl_out / f"{tile_stem}.txt"
                        with open(tile_lbl_path, "w", encoding="utf-8") as handle:
                            for cls_id, x, y, w, h in tile_boxes_list:
                                handle.write(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        tile_labels += 1
                        tile_boxes += len(tile_boxes_list)

        return {"images": tile_images, "labels": tile_labels, "boxes": tile_boxes}

    def run(self) -> dict:
        print("\n" + "#" * 72)
        print("# STEP 0: REBUILD TILED DATASET FROM RAW")
        print("#" * 72)
        print(f"[INFO] Raw data dir: {self.data_dir}")
        print(f"[INFO] Output tiled dir: {self.output_dir}")

        if not self.defect_dir.exists():
            raise FileNotFoundError(f"Missing defect image dir: {self.defect_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Missing yolo label dir: {self.label_dir}")

        raw_samples = self._collect_raw_samples()
        split_samples = self._split_raw_samples(raw_samples)
        self._prepare_output()

        split_stats: dict[str, dict[str, int]] = {}
        for split in ("train", "val", "test"):
            stats = self._tile_and_save(split, split_samples[split])
            stats["raw_samples"] = len(split_samples[split])
            split_stats[split] = stats
            print(
                f"[{split}] raw={stats['raw_samples']} tiled_images={stats['images']} "
                f"labels={stats['labels']} boxes={stats['boxes']}"
            )

        yaml_path = ensure_dataset_yaml(self.output_dir)
        summary = {
            "raw_sample_count": len(raw_samples),
            "split_raw_counts": {k: len(v) for k, v in split_samples.items()},
            "split_tiled_stats": split_stats,
            "dataset_yaml": str(yaml_path),
        }
        summary_path = self.output_dir / "rebuild_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        print("\n" + "=" * 72)
        print("[SUCCESS] Rebuilt tiled dataset.")
        print(f"[SAVED] {summary_path}")
        print("=" * 72)
        return summary


def main() -> int:
    TiledDatasetRebuilder().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
