#!/usr/bin/env python
from __future__ import annotations

"""
PF-MDD Baseline Training Script - Optimized for Wide-Format Images
YOLOv8s with Custom Configuration for Plastic Film Defect Detection
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from ultralytics import YOLO
from pathlib import Path
import json
import time
from datetime import datetime
import yaml

from project_config import (
    BALANCED_TILED_DATASET_DIR,
    EXPANDED_TILED_DATASET_DIR,
    REFINED_TILED_DATASET_DIR,
    TRAINED_MODELS_DIR,
    TRAINING_REPORTS_DIR,
    TRAINING_RESULTS_DIR,
    TILED_DATASET_DIR,
    YOLO_CLASS_NAMES,
    dataset_has_required_layout,
    ensure_dataset_yaml,
    ensure_project_dirs,
    preferred_training_dataset_dir,
)


def setup_environment():
    """Ensure all required directories exist"""
    ensure_project_dirs()
    for directory in (TRAINING_RESULTS_DIR, TRAINED_MODELS_DIR, TRAINING_REPORTS_DIR):
        print(f"[OK] {directory}")


def _dataset_root_from_yaml(dataset_yaml: Path, data_config: dict | None = None) -> Path:
    if data_config is None:
        with open(dataset_yaml, "r", encoding="utf-8") as handle:
            data_config = yaml.safe_load(handle)

    path_value = str(data_config.get("path", "")).strip()
    dataset_root = Path(path_value) if path_value else dataset_yaml.parent
    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml.parent / dataset_root).resolve(strict=False)
    return dataset_root


def validate_dataset_config(dataset_yaml: Path) -> None:
    """Fail fast if the dataset YAML and labels are inconsistent."""
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    with open(dataset_yaml, "r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    dataset_root = _dataset_root_from_yaml(dataset_yaml, data_config)

    nc = int(data_config["nc"])
    if nc != len(YOLO_CLASS_NAMES):
        raise ValueError(
            f"Dataset YAML reports nc={nc}, but the project expects {len(YOLO_CLASS_NAMES)} defect classes."
        )

    labels_root = dataset_root / "labels"
    if not labels_root.exists():
        raise FileNotFoundError(f"Dataset labels directory not found: {labels_root}")

    invalid_labels = []
    label_files = list(labels_root.rglob("*.txt"))
    if not label_files:
        raise ValueError(f"No label files found under {labels_root}")

    for label_path in label_files:
        with open(label_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                class_id = int(line.split()[0])
                if class_id < 0 or class_id >= nc:
                    invalid_labels.append((label_path.name, line_number, class_id))
                    if len(invalid_labels) >= 5:
                        break
        if len(invalid_labels) >= 5:
            break

    if invalid_labels:
        details = ", ".join(f"{name}:{line} -> {class_id}" for name, line, class_id in invalid_labels)
        raise ValueError(f"Found out-of-range class IDs in YOLO labels: {details}")


def resolve_training_dataset_yaml(overrides: dict | None = None) -> Path:
    """Resolve dataset YAML path with override support and tiled-dataset preference."""
    explicit = None
    if overrides:
        explicit = overrides.get("dataset_yaml")
        if explicit is None:
            explicit = overrides.get("data")

    if explicit is not None:
        candidate = Path(str(explicit))
        if candidate.suffix.lower() == ".yaml":
            return ensure_dataset_yaml(candidate.parent, candidate)
        return ensure_dataset_yaml(candidate)

    dataset_dir = preferred_training_dataset_dir()
    return ensure_dataset_yaml(dataset_dir)


def pick_device(preferred: str | None = None) -> str:
    """选择训练设备。

    规则：
      1) 显式传入 preferred 时优先使用（例如 "0"、"cpu"）
      2) 环境变量 YOLO_DEVICE 优先
      3) 有可用 CUDA 时用 "0"，否则退回 "cpu"

    原实现把 device 写死为 "cpu"，在有 GPU 的机器上会白白慢几十倍，
    也让 FPS 指标无法测量。
    """
    import os

    if preferred:
        return str(preferred)
    env_device = os.environ.get("YOLO_DEVICE", "").strip()
    if env_device:
        return env_device
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[OK] 检测到 GPU: {name}，使用 device=0")
            return "0"
    except Exception as exc:  # torch 未安装或驱动异常
        print(f"[WARN] CUDA 检测失败（{exc.__class__.__name__}），退回 CPU")
    print("[WARN] 未检测到可用 GPU，使用 CPU（训练会非常慢）")
    return "cpu"


def train_baseline_yolov8(overrides=None, model_weights='yolov8s.pt'):
    """
    Train baseline YOLOv8s model on PF-MDD dataset
    
    Model: YOLOv8s (Small - good balance of speed/accuracy)
    Dataset: PF-MDD (Plastic Film Multi-class Defect Detection)
    Input: 4096x256 images (processed to 640x640 tiles)
    Classes: 12 defect classes
    """
    
    print("\n" + "="*70)
    print(f"PF-MDD BASELINE TRAINING - {model_weights}")
    print("="*70)
    
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup directories
    setup_environment()
    
    # Load pre-trained YOLOv8 model (COCO pretrained)
    print(f"\n[STEP 1] Loading pre-trained model: {model_weights}...")
    model = YOLO(model_weights)
    
    print("[INFO] Model loaded successfully")
    print(f"  Architecture: {model_weights}")
    print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Training configuration optimized for PF-MDD
    override_dict = dict(overrides) if overrides else {}
    dataset_yaml = resolve_training_dataset_yaml(override_dict)
    dataset_root = _dataset_root_from_yaml(dataset_yaml)
    resolved_root = dataset_root.resolve(strict=False)
    if resolved_root == BALANCED_TILED_DATASET_DIR.resolve(strict=False):
        print("[INFO] Using class-balanced tiled dataset for training (best recall-oriented choice).")
    elif resolved_root == REFINED_TILED_DATASET_DIR.resolve(strict=False):
        print("[INFO] Using refined tiled dataset for training (re-split + minority-boosted train set).")
    elif resolved_root == EXPANDED_TILED_DATASET_DIR.resolve(strict=False):
        print("[INFO] Using expanded tiled dataset for training (higher positive sample density).")
    elif resolved_root == TILED_DATASET_DIR.resolve(strict=False):
        print("[INFO] Using tiled dataset for training (recommended for tiny defects).")
    elif dataset_has_required_layout(TILED_DATASET_DIR):
        print("[WARNING] Tiled dataset is available but not selected; training quality may be lower.")
    print(f"[INFO] Dataset config: {dataset_yaml}")
    print(f"[INFO] Dataset root: {dataset_root}")
    validate_dataset_config(dataset_yaml)

    print("\n[STEP 2] Starting training...")
    print("\nTraining Configuration:")
    print("  " + "-"*50)
    config = {
        # Data configuration
        'data': str(dataset_yaml),
        
        # Training hyperparameters
        'epochs': 200,
        'batch': 8,
        'imgsz': 640,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'workers': 4,
        'device': pick_device(override_dict.get('device')),
        
        # Optimizer settings
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Augmentation settings (optimized for defect detection)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,
        'perspective': 0.001,
        'flipud': 0.5,
        'fliplr': 0.5,
        'bgr': 0.0,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.1,
        'erasing': 0.4,
        'crop_fraction': 1.0,
        
        # YOLOv8 损失权重系数（box=框回归, cls=分类, dfl=分布焦点损失）
        # 注意：这不是 Focal Loss。YOLOv8 默认分类损失已含 DFL，
        # 若要引入 Focal Loss 需修改 ultralytics 的 loss 实现。
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Project settings
        'project': str(TRAINING_RESULTS_DIR),
        'name': f'{Path(model_weights).stem}_{timestamp}',
        'exist_ok': True,
        'pretrained': True,
        'resume': False,
        
        # Validation settings
        'val': True,
        'plots': True,
        
        # Verbose output
        'verbose': True
    }

    # `dataset_yaml` and `data` are consumed above to build a normalized YAML path.
    override_dict.pop("dataset_yaml", None)
    override_dict.pop("data", None)
    if override_dict:
        config.update(override_dict)
    
    # Print configuration
    for key, value in config.items():
        if isinstance(value, (int, float)):
            print(f"  {key:20s}: {value}")
        else:
            print(f"  {key:20s}: {str(value)[:40]}")
    
    print("  " + "-"*50)
    
    try:
        # Start training
        try:
            results = model.train(**config)
        except PermissionError:
            # Some restricted Windows environments block ultralytics' default ThreadPool backend.
            from concurrent.futures import ThreadPoolExecutor
            import ultralytics.data.dataset as yolo_dataset_module

            class SafeThreadPool:
                """ThreadPool fallback that avoids multiprocessing pipes."""

                def __init__(self, processes=None, *args, **kwargs):
                    max_workers = max(1, int(processes or 1))
                    self._executor = ThreadPoolExecutor(max_workers=max_workers)

                def imap(self, func, iterable):
                    futures = [self._executor.submit(func, item) for item in iterable]
                    for future in futures:
                        yield future.result()

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    self._executor.shutdown(wait=True)

            print("[WARNING] Permission error with default data loader backend. Retrying with safe thread pool...")
            yolo_dataset_module.ThreadPool = SafeThreadPool
            results = model.train(**config)
        
        training_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("[TRAINING COMPLETED]")
        print("="*70)
        print(f"\nTraining Time: {training_time/60:.1f} minutes ({training_time:.1f} seconds)")
        print(f"\nResults saved to: {results.save_dir}")
        
        # Save training summary
        save_training_summary(config, results, training_time, timestamp, model_weights)
        
        return results, model, timestamp
        
    except Exception as e:
        print(f"\n[ERROR] Training failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def save_training_summary(config, results, training_time, timestamp, model_weights='yolov8s.pt'):
    """Save comprehensive training summary"""
    
    summary = {
        'training_info': {
            'timestamp': timestamp,
            'model_architecture': Path(model_weights).stem,
            'dataset': 'PF-MDD',
            'total_epochs': config['epochs'],
            'actual_epochs_trained': getattr(results, 'epochs', config['epochs']),
            'training_time_seconds': round(training_time, 2),
            'best_epoch': getattr(results, 'best_epoch', 'N/A'),
        },
        
        'configuration': {
            k: v for k, v in config.items() 
            if k not in ['project', 'name', 'exist_ok', 'data']
        },
        
        'output_paths': {
            'results_dir': str(getattr(results, 'save_dir', 'N/A')),
            'weights_best': f"{getattr(results, 'save_dir', '')}/weights/best.pt",
            'weights_last': f"{getattr(results, 'save_dir', '')}/weights/last.pt",
        },
        
        'notes': [
            "Baseline model trained on PF-MDD dataset",
            "Image resolution: 4096x256 -> processed to 640x640",
            "Class loss weighting tuned for long-tail defect classes",
            "Full augmentation pipeline applied"
        ]
    }
    
    summary_path = TRAINING_RESULTS_DIR / f'training_summary_{timestamp}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SAVED] Training summary: {summary_path}")


if __name__ == '__main__':
    print("#"*70)
    print("# PF-MDD BASELINE MODEL TRAINING")
    print("# YOLOv8s for Plastic Film Defect Detection")
    print("#"*70)
    
    results, model, timestamp = train_baseline_yolov8()
    
    if results is not None:
        print("\n" + "#"*70)
        print("# SUCCESS! Baseline training completed.")
        print("# Next step: Run evaluation script")
        print("#   > python evaluate_model.py")
        print("#"*70)
    else:
        print("\nTraining failed. Check error messages above.")
        sys.exit(1)
