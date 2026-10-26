"""
PF-MDD Mask to YOLO Format Converter
Converts pixel-level masks to YOLO format bounding boxes
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

from project_config import (
    BASE_DIR,
    DATA_DIR,
    DEFECT_TYPE_TO_YOLO_INDEX,
    YOLO_INDEX_TO_DEFECT_TYPE,
    resolve_data_path,
)

sys.stdout.reconfigure(encoding='utf-8')

class MaskToYOLOConverter:
    def __init__(self, data_root, output_dir=None):
        self.data_root = Path(data_root)
        self.mask_dir = self.data_root / "Mask_images" / "Mask_images"
        self.defect_dir = self.data_root / "Defect_images" / "Defect_images"
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.data_root / "yolo_labels"
        
        self.output_dir.mkdir(exist_ok=True)
        
        self.conversion_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_boxes': 0,
            'boxes_per_class': defaultdict(int),
            'errors': []
        }
    
    def mask_to_yolo_boxes(self, mask_path, class_id, image_shape):
        """
        Convert a single mask image to YOLO format bounding boxes
        
        Args:
            mask_path: path to mask image
            class_id: defect class ID
            image_shape: (height, width) of original image
            
        Returns:
            list of YOLO format boxes: [class_id, x_center, y_center, w, h]
        """
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                raise ValueError(f"Cannot read mask: {mask_path}")
            
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(
                binary_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            boxes = []
            h, w = image_shape[:2]
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                x, y, bw, bh = cv2.boundingRect(contour)
                
                if bw < 3 or bh < 3:
                    continue
                
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                norm_w = bw / w
                norm_h = bh / h
                
                box = [class_id, x_center, y_center, norm_w, norm_h]
                boxes.append(box)
                self.conversion_stats['boxes_per_class'][class_id] += 1
            
            return boxes
            
        except Exception as e:
            self.conversion_stats['errors'].append({
                'file': str(mask_path),
                'error': str(e)
            })
            return []
    
    def get_image_shape(self, image_path):
        """Get image dimensions"""
        img = cv2.imread(str(image_path))
        if img is not None:
            return img.shape[:2]
        return (256, 4096)  # default PF-MDD resolution
    
    def convert_sample(self, sample_id, mask_paths, class_id, image_path):
        """
        Convert all masks for a single sample to YOLO format
        
        Args:
            sample_id: sample identifier
            mask_paths: list of mask file paths
            class_id: defect type ID
            image_path: corresponding defect image path
            
        Returns:
            list of all bounding boxes for this sample
        """
        image_shape = self.get_image_shape(image_path)
        all_boxes = []
        
        for mask_path in mask_paths:
            boxes = self.mask_to_yolo_boxes(mask_path, class_id, image_shape)
            all_boxes.extend(boxes)
        
        return all_boxes
    
    def save_yolo_label(self, sample_id, boxes, output_filename=None):
        """
        Save YOLO format labels to text file
        
        Format: class_id x_center y_center width height
        All values normalized to [0, 1]
        """
        if output_filename is None:
            output_filename = f"{sample_id}.txt"
        
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w') as f:
            for box in boxes:
                line = f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                f.write(line)
        
        return output_path
    
    def process_all_defect_samples(self, dataset_info_file=None):
        """
        Process all defect samples and convert their masks to YOLO format
        """
        print("\n" + "="*70)
        print("[STEP 1] Loading Dataset Information")
        print("="*70)
        
        if dataset_info_file and Path(dataset_info_file).exists():
            with open(dataset_info_file, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
            print(f"[OK] Loaded from: {dataset_info_file}")
        else:
            print("[INFO] Scanning directories...")
            dataset_info = self._scan_defect_samples()
        
        defect_samples = dataset_info.get('defect_samples', [])
        print(f"\nFound {len(defect_samples)} defect samples to process")
        
        print("\n" + "="*70)
        print("[STEP 2] Converting Masks to YOLO Format")
        print("="*70)
        
        for idx, sample in enumerate(defect_samples, 1):
            sample_id = sample['id']
            mask_paths = [resolve_data_path(p) for p in sample.get('mask_paths', [])]
            raw_defect_type = sample.get('defect_type', 0)
            if raw_defect_type not in DEFECT_TYPE_TO_YOLO_INDEX:
                self.conversion_stats['errors'].append({
                    'file': sample_id,
                    'error': f'Unsupported defect type: {raw_defect_type}'
                })
                print(f"[SKIP] {sample_id}: unsupported defect type {raw_defect_type}")
                continue

            class_id = DEFECT_TYPE_TO_YOLO_INDEX[raw_defect_type]
            image_path = resolve_data_path(sample['image_path'])
            
            if not mask_paths:
                print(f"[SKIP] {sample_id}: No masks found")
                continue
            
            self.conversion_stats['total_processed'] += 1
            
            boxes = self.convert_sample(sample_id, mask_paths, class_id, image_path)
            
            if boxes:
                self.save_yolo_label(sample_id, boxes)
                self.conversion_stats['successful'] += 1
                self.conversion_stats['total_boxes'] += len(boxes)
                
                if idx % 20 == 0 or idx <= 5:
                    print(
                        f"[{idx:3d}] {sample_id}: {len(boxes)} boxes "
                        f"(raw defect {raw_defect_type} -> yolo class {class_id})"
                    )
            else:
                self.conversion_stats['failed'] += 1
                print(f"[FAIL] {sample_id}: Could not generate boxes")
        
        self._print_conversion_summary()
        self._save_conversion_stats()
        
        return self.conversion_stats
    
    def _scan_defect_samples(self):
        """Scan defect directory if no info file available"""
        samples = []
        for img_path in sorted(self.defect_dir.glob("*.png")):
            parts = img_path.stem.split("_")
            if len(parts) >= 2:
                sample_id = img_path.stem
                class_id = int(parts[1])
                
                mask_paths = []
                base_mask = self.mask_dir / f"{sample_id}_mask.png"
                if base_mask.exists():
                    mask_paths.append(base_mask)
                
                mask1 = self.mask_dir / f"{sample_id}_mask1.png"
                if mask1.exists():
                    mask_paths.append(mask1)
                
                mask2 = self.mask_dir / f"{sample_id}_mask2.png"
                if mask2.exists():
                    mask_paths.append(mask2)
                
                samples.append({
                    'id': sample_id,
                    'image_path': str(img_path),
                    'mask_paths': [str(m) for m in mask_paths],
                    'defect_type': class_id
                })
        
        return {'defect_samples': samples}
    
    def _print_conversion_summary(self):
        """Print conversion statistics"""
        stats = self.conversion_stats
        
        print("\n" + "="*70)
        print("[CONVERSION SUMMARY]")
        print("="*70)
        print(f"\nTotal samples processed: {stats['total_processed']}")
        print(f"Successful conversions: {stats['successful']}")
        print(f"Failed conversions: {stats['failed']}")
        print(f"Total bounding boxes generated: {stats['total_boxes']}")
        
        print(f"\nBoxes per defect class:")
        for cls_id in sorted(stats['boxes_per_class'].keys()):
            count = stats['boxes_per_class'][cls_id]
            bar = "#" * max(1, count // 2)
            print(f"  Class {cls_id:03d}: {count:4d} boxes {bar}")
        
        if stats['errors']:
            print(f"\nErrors encountered: {len(stats['errors'])}")
            for err in stats['errors'][:3]:
                print(f"  - {err['file']}: {err['error']}")
    
    def _save_conversion_stats(self):
        """Save conversion statistics to JSON"""
        stats_file = self.output_dir / "conversion_stats.json"
        stats_to_save = dict(self.conversion_stats)
        stats_to_save['boxes_per_class'] = dict(stats_to_save['boxes_per_class'])
        stats_to_save['class_mapping'] = {
            str(class_idx): defect_type for class_idx, defect_type in YOLO_INDEX_TO_DEFECT_TYPE.items()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVED] Conversion stats saved to: {stats_file}")

def create_yolo_dataset_structure(output_base_dir):
    """Create standard YOLO dataset directory structure"""
    base = Path(output_base_dir)
    
    dirs = {
        'images/train': base / 'images' / 'train',
        'images/val': base / 'images' / 'val',
        'images/test': base / 'images' / 'test',
        'labels/train': base / 'labels' / 'train',
        'labels/val': base / 'labels' / 'val',
        'labels/test': base / 'labels' / 'test'
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"[CREATED] {name}")
    
    return base

def main():
    data_root = DATA_DIR
    dataset_info_file = data_root / "pf_mdd_dataset_info.json"
    
    print("\n" + "#"*70)
    print("# PF-MDD Mask to YOLO Converter")
    print("# Converting pixel-level masks to YOLO format")
    print("#"*70)
    
    converter = MaskToYOLOConverter(data_root)
    
    converter.process_all_defect_samples(dataset_info_file)
    
    print("\n" + "="*70)
    print("[INFO] Creating YOLO dataset structure...")
    print("="*70)
    
    yolo_dataset_dir = BASE_DIR / "yolo_dataset"
    create_yolo_dataset_structure(yolo_dataset_dir)
    
    print("\n" + "="*70)
    print("[SUCCESS] Mask to YOLO conversion completed!")
    print(f"Labels saved to: {converter.output_dir}")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
