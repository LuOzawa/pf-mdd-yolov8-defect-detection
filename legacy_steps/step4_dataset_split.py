"""
PF-MDD Dataset Splitting Script
Splits augmented dataset into train/val/test sets for YOLO training
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

from project_config import DATA_DIR, TILED_DATASET_DIR, YOLO_CLASS_NAMES, dataset_yaml_text

sys.stdout.reconfigure(encoding='utf-8')


class DatasetSplitter:
    """
    Split PF-MDD dataset into train/validation/test sets
    Ensures balanced class distribution across splits
    """

    def __init__(self, input_dir, output_dir, split_ratios=(0.7, 0.15, 0.15)):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.split_ratios = split_ratios

        self.input_images = self.input_dir / "images"
        self.input_labels = self.input_dir / "labels"

        self.split_stats = {
            'split_ratios': {
                'train': split_ratios[0],
                'val': split_ratios[1],
                'test': split_ratios[2]
            },
            'total_samples': 0,
            'train': {'count': 0, 'classes': defaultdict(int)},
            'val': {'count': 0, 'classes': defaultdict(int)},
            'test': {'count': 0, 'classes': defaultdict(int)}
        }

    def create_directory_structure(self):
        """Create YOLO dataset directory structure"""
        print("\n" + "="*70)
        print("[STEP 1] Creating Directory Structure")
        print("="*70)

        dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "images" / "test",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val",
            self.output_dir / "labels" / "test"
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            print(f"[OK] {d.relative_to(self.output_dir)}")

    def load_samples_by_class(self):
        """Load all samples and group by class"""
        print("\n" + "="*70)
        print("[STEP 2] Loading and Grouping Samples by Class")
        print("="*70)

        samples_by_class = defaultdict(list)

        if not self.input_images.exists():
            print(f"[ERROR] Input images directory not found: {self.input_images}")
            return samples_by_class

        image_files = list(self.input_images.glob("*.png"))
        print(f"Found {len(image_files)} images")

        for img_path in image_files:
            label_path = self.input_labels / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    first_box = lines[0].strip().split()
                    if len(first_box) >= 5:
                        class_id = int(first_box[0])
                        samples_by_class[class_id].append({
                            'image': img_path,
                            'label': label_path,
                            'stem': img_path.stem
                        })

        print(f"\nClass distribution:")
        for class_id in sorted(samples_by_class.keys()):
            count = len(samples_by_class[class_id])
            print(f"  Class {class_id:03d}: {count} samples")

        return samples_by_class

    def stratified_split(self, samples_by_class):
        """Perform stratified split to maintain class balance"""
        print("\n" + "="*70)
        print("[STEP 3] Performing Stratified Split")
        print("="*70)

        train_ratio, val_ratio, test_ratio = self.split_ratios
        splits = {'train': [], 'val': [], 'test': []}

        random.seed(42)

        for class_id, samples in samples_by_class.items():
            random.shuffle(samples)
            n_samples = len(samples)

            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)

            train_samples = samples[:n_train]
            val_samples = samples[n_train:n_train + n_val]
            test_samples = samples[n_train + n_val:]

            splits['train'].extend(train_samples)
            splits['val'].extend(val_samples)
            splits['test'].extend(test_samples)

            self.split_stats['train']['classes'][class_id] = len(train_samples)
            self.split_stats['val']['classes'][class_id] = len(val_samples)
            self.split_stats['test']['classes'][class_id] = len(test_samples)

            print(f"  Class {class_id:03d}: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

        self.split_stats['train']['count'] = len(splits['train'])
        self.split_stats['val']['count'] = len(splits['val'])
        self.split_stats['test']['count'] = len(splits['test'])
        self.split_stats['total_samples'] = sum(len(s) for s in splits.values())

        print(f"\nTotal split:")
        print(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/self.split_stats['total_samples']*100:.1f}%)")
        print(f"  Val:   {len(splits['val'])} samples ({len(splits['val'])/self.split_stats['total_samples']*100:.1f}%)")
        print(f"  Test:  {len(splits['test'])} samples ({len(splits['test'])/self.split_stats['total_samples']*100:.1f}%)")

        return splits

    def copy_files_to_splits(self, splits):
        """Copy files to their respective split directories"""
        print("\n" + "="*70)
        print("[STEP 4] Copying Files to Split Directories")
        print("="*70)

        for split_name, samples in splits.items():
            print(f"\nCopying {split_name} set ({len(samples)} samples)...")

            img_dir = self.output_dir / "images" / split_name
            label_dir = self.output_dir / "labels" / split_name

            for idx, sample in enumerate(samples, 1):
                shutil.copy(sample['image'], img_dir / sample['image'].name)
                shutil.copy(sample['label'], label_dir / sample['label'].name)

                if idx % 50 == 0:
                    print(f"  [{idx}/{len(samples)}] copied...")

            print(f"  [DONE] {split_name} set complete")

    def create_yaml_config(self):
        """Create YOLO dataset configuration file"""
        print("\n" + "="*70)
        print("[STEP 5] Creating YOLO Configuration File")
        print("="*70)

        yaml_path = self.output_dir / "pf_mdd.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(dataset_yaml_text(self.output_dir))

        print(f"[SAVED] Configuration file: {yaml_path}")

    def save_split_statistics(self):
        """Save split statistics to JSON"""
        print("\n" + "="*70)
        print("[STEP 6] Saving Split Statistics")
        print("="*70)

        stats_to_save = dict(self.split_stats)
        for split in ['train', 'val', 'test']:
            stats_to_save[split]['classes'] = dict(stats_to_save[split]['classes'])
        stats_to_save['class_mapping'] = {
            str(class_idx): class_name for class_idx, class_name in YOLO_CLASS_NAMES.items()
        }

        stats_file = self.output_dir / "split_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, indent=2, ensure_ascii=False)

        print(f"[SAVED] Statistics file: {stats_file}")

        print("\n[SPLIT SUMMARY]")
        print(f"  Total samples: {self.split_stats['total_samples']}")
        print(f"  Train: {self.split_stats['train']['count']} ({self.split_stats['train']['count']/self.split_stats['total_samples']*100:.1f}%)")
        print(f"  Val:   {self.split_stats['val']['count']} ({self.split_stats['val']['count']/self.split_stats['total_samples']*100:.1f}%)")
        print(f"  Test:  {self.split_stats['test']['count']} ({self.split_stats['test']['count']/self.split_stats['total_samples']*100:.1f}%)")

    def split_dataset(self):
        """Main workflow to split the dataset"""
        self.create_directory_structure()
        samples_by_class = self.load_samples_by_class()

        if not samples_by_class:
            print("[ERROR] No samples found to split")
            return False

        splits = self.stratified_split(samples_by_class)
        self.copy_files_to_splits(splits)
        self.create_yaml_config()
        self.save_split_statistics()

        return True


def main():
    input_dir = DATA_DIR / "augmented_dataset"
    output_dir = TILED_DATASET_DIR

    print("\n" + "#"*70)
    print("# PF-MDD Dataset Splitting Tool")
    print("# Creating train/val/test splits for YOLO training")
    print("#"*70)

    splitter = DatasetSplitter(
        input_dir=input_dir,
        output_dir=output_dir,
        split_ratios=(0.7, 0.15, 0.15)
    )

    success = splitter.split_dataset()

    if success:
        print("\n" + "="*70)
        print("[SUCCESS] Dataset splitting completed!")
        print(f"Output directory: {output_dir}")
        print("\nNext step: Run training script")
        print("  > python train_baseline.py")
        print("="*70)
    else:
        print("\n[ERROR] Dataset splitting failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
