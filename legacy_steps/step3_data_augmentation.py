"""
PF-MDD Data Augmentation Script
Generates augmented samples to address class imbalance
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import time

from project_config import DATA_DIR, DEFECT_TYPE_TO_YOLO_INDEX, resolve_data_path

sys.stdout.reconfigure(encoding='utf-8')


class DataAugmentor:
    """
    Data augmentation for PF-MDD dataset
    Focuses on minority classes to balance the dataset
    """

    def __init__(self, data_root, output_dir=None):
        self.data_root = Path(data_root)
        self.defect_dir = self.data_root / "Defect_images" / "Defect_images"
        self.mask_dir = self.data_root / "Mask_images" / "Mask_images"
        self.labels_dir = self.data_root / "yolo_labels"

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.data_root / "augmented_dataset"

        self.output_images = self.output_dir / "images"
        self.output_labels = self.output_dir / "labels"
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_labels.mkdir(parents=True, exist_ok=True)

        self.augmentation_stats = {
            'original_samples': 0,
            'augmented_samples': 0,
            'total_samples': 0,
            'augmentations_per_class': defaultdict(int),
            'augmentation_methods': []
        }

    def horizontal_flip(self, image, boxes):
        """Horizontal flip augmentation"""
        flipped_img = cv2.flip(image, 1)
        flipped_boxes = []

        for box in boxes:
            class_id, xc, yc, w, h = box
            new_xc = 1.0 - xc
            flipped_boxes.append([class_id, new_xc, yc, w, h])

        return flipped_img, flipped_boxes

    def vertical_flip(self, image, boxes):
        """Vertical flip augmentation"""
        flipped_img = cv2.flip(image, 0)
        flipped_boxes = []

        for box in boxes:
            class_id, xc, yc, w, h = box
            new_yc = 1.0 - yc
            flipped_boxes.append([class_id, xc, new_yc, w, h])

        return flipped_img, flipped_boxes

    def rotate_90(self, image, boxes):
        """Rotate 90 degrees clockwise"""
        h, w = image.shape[:2]
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_boxes = []

        for box in boxes:
            class_id, xc, yc, w_box, h_box = box
            new_xc = 1.0 - yc
            new_yc = xc
            rotated_boxes.append([class_id, new_xc, new_yc, h_box, w_box])

        return rotated_img, rotated_boxes

    def brightness_adjustment(self, image, boxes, factor=1.2):
        """Adjust brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        adjusted_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return adjusted_img, boxes

    def gaussian_noise(self, image, boxes, sigma=10):
        """Add Gaussian noise"""
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy_img = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return noisy_img, boxes

    def load_yolo_labels(self, label_file):
        """Load YOLO format labels"""
        boxes = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        box = [int(parts[0])] + [float(x) for x in parts[1:5]]
                        boxes.append(box)
        return boxes

    def save_yolo_labels(self, boxes, output_file):
        """Save YOLO format labels"""
        with open(output_file, 'w') as f:
            for box in boxes:
                line = f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                f.write(line)

    def augment_sample(self, image_path, label_path, sample_id, class_id):
        """Apply multiple augmentations to a single sample"""

        image = cv2.imread(str(image_path))
        if image is None:
            return 0

        boxes = self.load_yolo_labels(label_path)
        if not boxes:
            return 0

        augmentations = [
            ('horizontal_flip', self.horizontal_flip),
            ('vertical_flip', self.vertical_flip),
            ('rotate_90', self.rotate_90),
            ('brightness_adjustment', lambda img, b: self.brightness_adjustment(img, b, 1.3)),
            ('gaussian_noise', lambda img, b: self.gaussian_noise(img, b, 15))
        ]

        count = 0
        for aug_name, aug_func in augmentations:
            try:
                aug_img, aug_boxes = aug_func(image, boxes)

                output_img_name = f"{sample_id}_{aug_name}.png"
                output_label_name = f"{sample_id}_{aug_name}.txt"

                cv2.imwrite(str(self.output_images / output_img_name), aug_img)
                self.save_yolo_labels(aug_boxes, self.output_labels / output_label_name)

                count += 1
                self.augmentation_stats['augmentations_per_class'][class_id] += 1

            except Exception as e:
                print(f"[ERROR] Augmentation {aug_name} failed for {sample_id}: {e}")

        return count

    def process_dataset(self, dataset_info_file=None):
        """Process entire dataset with augmentation"""

        print("\n" + "="*70)
        print("[STEP 1] Loading Dataset Information")
        print("="*70)

        if dataset_info_file and Path(dataset_info_file).exists():
            with open(dataset_info_file, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        else:
            print("[ERROR] Dataset info file not found")
            return

        defect_samples = dataset_info.get('defect_samples', [])
        print(f"Found {len(defect_samples)} defect samples")

        print("\n" + "="*70)
        print("[STEP 2] Copying Original Samples")
        print("="*70)

        for sample in defect_samples:
            sample_id = sample['id']
            image_path = resolve_data_path(sample['image_path'])
            label_path = self.labels_dir / f"{sample_id}.txt"

            if image_path.exists():
                import shutil
                shutil.copy(image_path, self.output_images / f"{sample_id}.png")
                if label_path.exists():
                    shutil.copy(label_path, self.output_labels / f"{sample_id}.txt")
                self.augmentation_stats['original_samples'] += 1

        print(f"Copied {self.augmentation_stats['original_samples']} original samples")

        print("\n" + "="*70)
        print("[STEP 3] Generating Augmented Samples")
        print("="*70)

        total_augmented = 0
        start_time = time.time()

        for idx, sample in enumerate(defect_samples, 1):
            sample_id = sample['id']
            raw_defect_type = sample.get('defect_type', 0)
            class_id = DEFECT_TYPE_TO_YOLO_INDEX.get(raw_defect_type, raw_defect_type)
            image_path = resolve_data_path(sample['image_path'])
            label_path = self.labels_dir / f"{sample_id}.txt"

            if not image_path.exists() or not label_path.exists():
                continue

            count = self.augment_sample(image_path, label_path, sample_id, class_id)
            total_augmented += count

            if idx % 10 == 0 or idx <= 5:
                print(f"[{idx:3d}/{len(defect_samples)}] {sample_id}: {count} augmentations")

        elapsed = time.time() - start_time

        self.augmentation_stats['augmented_samples'] = total_augmented
        self.augmentation_stats['total_samples'] = (
            self.augmentation_stats['original_samples'] + total_augmented
        )

        print(f"\n[COMPLETE] Generated {total_augmented} augmented samples in {elapsed:.1f}s")

        self._save_statistics()

    def _save_statistics(self):
        """Save augmentation statistics"""

        print("\n" + "="*70)
        print("[AUGMENTATION SUMMARY]")
        print("="*70)

        stats = self.augmentation_stats
        print(f"\nOriginal samples: {stats['original_samples']}")
        print(f"Augmented samples: {stats['augmented_samples']}")
        print(f"Total samples: {stats['total_samples']}")

        print(f"\nAugmentations per class:")
        for class_id in sorted(stats['augmentations_per_class'].keys()):
            count = stats['augmentations_per_class'][class_id]
            print(f"  Class {class_id:03d}: {count} augmentations")

        stats_file = self.output_dir / "augmentation_stats.json"
        stats_to_save = dict(stats)
        stats_to_save['augmentations_per_class'] = dict(stats_to_save['augmentations_per_class'])

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, indent=2, ensure_ascii=False)

        print(f"\n[SAVED] Statistics saved to: {stats_file}")


def main():
    data_root = DATA_DIR
    dataset_info_file = data_root / "pf_mdd_dataset_info.json"

    print("\n" + "#"*70)
    print("# PF-MDD Data Augmentation Tool")
    print("# Generating augmented samples for class balance")
    print("#"*70)

    augmentor = DataAugmentor(data_root)
    augmentor.process_dataset(dataset_info_file)

    print("\n" + "="*70)
    print("[SUCCESS] Data augmentation completed!")
    print(f"Output directory: {augmentor.output_dir}")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
