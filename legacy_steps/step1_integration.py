"""
PF-MDD Dataset Integration and Cleaning Script
Plastic Film Multi-class Defect Detection Dataset
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from project_config import DATA_DIR

sys.stdout.reconfigure(encoding='utf-8')

class PFMDDBase:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.defect_dir = self.data_root / "Defect_images" / "Defect_images"
        self.mask_dir = self.data_root / "Mask_images" / "Mask_images"
        self.nodefect_dir = self.data_root / "NODefect_images" / "NODefect_images"
        
        self.dataset_info = {
            'name': 'PF-MDD',
            'description': 'Plastic Film Multi-class Defect Detection Dataset',
            'resolution': (4096, 256),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'defect_samples': [],
            'nodefect_samples': [],
            'statistics': {}
        }
    
    def validate_directories(self):
        print("\n" + "="*70)
        print("[STEP 1] Validating Directory Structure")
        print("="*70)
        
        checks = {
            'Defect Images': self.defect_dir,
            'Mask Images': self.mask_dir,
            'No-Defect Images': self.nodefect_dir
        }
        
        for name, path in checks.items():
            exists = path.exists()
            status = "[OK]" if exists else "[MISSING]"
            print(f"{status} {name}: {path}")
            
            if exists:
                count = len(list(path.glob("*.png")))
                print(f"      -> Found {count} PNG files")
        
        return all(p.exists() for p in checks.values())
    
    def collect_defect_samples(self):
        print("\n" + "="*70)
        print("[STEP 2] Collecting Defect Samples")
        print("="*70)
        
        defect_samples = []
        defect_types = defaultdict(int)
        
        if not self.defect_dir.exists():
            return defect_samples
        
        for img_path in sorted(self.defect_dir.glob("*.png")):
            parts = img_path.stem.split("_")
            if len(parts) >= 2:
                sample_id = img_path.stem
                defect_type = int(parts[1])
                
                sample_info = {
                    'id': sample_id,
                    'image_path': str(img_path),
                    'mask_paths': [],
                    'defect_type': defect_type,
                    'has_defect': True
                }
                
                mask_files = self._find_masks(sample_id)
                sample_info['mask_paths'] = [str(m) for m in mask_files]
                
                defect_samples.append(sample_info)
                defect_types[defect_type] += 1
        
        print(f"\nTotal defect samples collected: {len(defect_samples)}")
        print("\nDefect type distribution:")
        for dtype in sorted(defect_types.keys()):
            bar = "#" * (defect_types[dtype] // 2)
            print(f"  Type {dtype:03d}: {defect_types[dtype]:3d} samples {bar}")
        
        self.dataset_info['defect_samples'] = defect_samples
        self.dataset_info['statistics']['defect_types'] = dict(defect_types)
        
        return defect_samples
    
    def _find_masks(self, sample_id):
        masks = []
        
        base_mask = self.mask_dir / f"{sample_id}_mask.png"
        if base_mask.exists():
            masks.append(base_mask)
        
        mask1 = self.mask_dir / f"{sample_id}_mask1.png"
        if mask1.exists():
            masks.append(mask1)
        
        mask2 = self.mask_dir / f"{sample_id}_mask2.png"
        if mask2.exists():
            masks.append(mask2)
        
        return masks
    
    def collect_nodefect_samples(self):
        print("\n" + "="*70)
        print("[STEP 3] Collecting No-Defect Samples")
        print("="*70)
        
        nodefect_samples = []
        
        if not self.nodefect_dir.exists():
            return nodefect_samples
        
        folder_counts = {}
        
        for folder in sorted(self.nodefect_dir.iterdir()):
            if folder.is_dir():
                images = list(folder.glob("*.png"))
                folder_counts[folder.name] = len(images)
                
                for img_path in sorted(images):
                    sample_info = {
                        'id': img_path.stem,
                        'image_path': str(img_path),
                        'source_folder': folder.name,
                        'defect_type': 0,
                        'has_defect': False
                    }
                    nodefect_samples.append(sample_info)
        
        print(f"\nTotal no-defect samples collected: {len(nodefect_samples)}")
        print(f"No-defect folders: {len(folder_counts)}")
        print("\nSamples per folder:")
        for folder, count in sorted(folder_counts.items()):
            print(f"  {folder}: {count} images")
        
        self.dataset_info['nodefect_samples'] = nodefect_samples
        self.dataset_info['statistics']['nodefect_folders'] = folder_counts
        
        return nodefect_samples
    
    def validate_data_integrity(self):
        print("\n" + "="*70)
        print("[STEP 4] Validating Data Integrity")
        print("="*70)
        
        issues = []
        
        print("\nChecking defect samples...")
        missing_masks = []
        for sample in self.dataset_info['defect_samples']:
            if not sample['mask_paths']:
                missing_masks.append(sample['id'])
        
        if missing_masks:
            issues.append(f"Missing masks for {len(missing_masks)} samples")
            print(f"[WARNING] {len(missing_masks)} samples without masks:")
            for mid in missing_masks[:5]:
                print(f"    - {mid}")
        else:
            print("[OK] All defect samples have corresponding masks")
        
        print("\nChecking file existence...")
        missing_files = []
        for sample in self.dataset_info['defect_samples']:
            if not Path(sample['image_path']).exists():
                missing_files.append(sample['image_path'])
        
        for sample in self.dataset_info['nodefect_samples']:
            if not Path(sample['image_path']).exists():
                missing_files.append(sample['image_path'])
        
        if missing_files:
            issues.append(f"{len(missing_files)} files not found")
            print(f"[ERROR] {len(missing_files)} files missing")
        else:
            print("[OK] All image files exist")
        
        multi_defect = [s for s in self.dataset_info['defect_samples'] 
                       if len(s['mask_paths']) > 1]
        if multi_defect:
            print(f"\n[INFO] Found {len(multi_defect)} samples with multiple defects:")
            for s in multi_defect:
                print(f"    - {s['id']}: {len(s['mask_paths'])} masks")
        
        self.dataset_info['statistics']['integrity_issues'] = issues
        return len(issues) == 0
    
    def generate_dataset_summary(self):
        print("\n" + "="*70)
        print("[FINAL] Dataset Summary Report")
        print("="*70)
        
        total_defect = len(self.dataset_info['defect_samples'])
        total_nodefect = len(self.dataset_info['nodefect_samples'])
        total = total_defect + total_nodefect
        
        summary = {
            'dataset_name': 'PF-MDD',
            'total_samples': total,
            'defect_samples': total_defect,
            'nodefect_samples': total_nodefect,
            'num_defect_types': len(self.dataset_info['statistics'].get('defect_types', {})),
            'resolution': '4096x256',
            'defect_ratio': f"{total_defect/total*100:.1f}%",
            'has_multi_defect_samples': any(
                len(s['mask_paths']) > 1 
                for s in self.dataset_info['defect_samples']
            )
        }
        
        print(f"\nDataset: {summary['dataset_name']}")
        print(f"Total Samples: {summary['total_samples']}")
        print(f"  - Defect: {summary['defect_samples']} ({summary['defect_ratio']})")
        print(f"  - No-Defect: {summary['nodefect_samples']}")
        print(f"Defect Types: {summary['num_defect_types']} categories")
        print(f"Resolution: {summary['resolution']}")
        print(f"Multi-defect cases: {'Yes' if summary['has_multi_defect_samples'] else 'No'}")
        
        self.dataset_info['summary'] = summary
        return summary
    
    def save_dataset_info(self, output_file="pf_mdd_dataset_info.json"):
        output_path = self.data_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset_info, f, ensure_ascii=False, indent=2)
        print(f"\n[SAVED] Dataset info saved to: {output_path}")
        return output_path

def main():
    data_root = str(DATA_DIR)
    
    print("\n" + "#"*70)
    print("# PF-MDD Dataset Integration and Cleaning Tool")
    print("# Plastic Film Multi-class Defect Detection")
    print("#"*70)
    
    dataset = PFMDDBase(data_root)
    
    if dataset.validate_directories():
        dataset.collect_defect_samples()
        dataset.collect_nodefect_samples()
        dataset.validate_data_integrity()
        dataset.generate_dataset_summary()
        dataset.save_dataset_info()
        
        print("\n" + "="*70)
        print("[SUCCESS] Dataset integration completed!")
        print("="*70)
    else:
        print("\n[ERROR] Directory validation failed. Please check paths.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
