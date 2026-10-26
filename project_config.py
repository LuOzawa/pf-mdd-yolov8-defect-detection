from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# Backward-compatible alias: default YOLO dataset now points to the tiled dataset.
YOLO_DATASET_DIR = BASE_DIR / "pfmd_tiled_dataset"
TILED_DATASET_DIR = BASE_DIR / "pfmd_tiled_dataset"
REFINED_TILED_DATASET_DIR = BASE_DIR / "pfmd_tiled_dataset_refined"
EXPANDED_TILED_DATASET_DIR = BASE_DIR / "pfmd_tiled_dataset_expanded"
BALANCED_TILED_DATASET_DIR = BASE_DIR / "pfmd_tiled_dataset_balanced"
TRAINING_RESULTS_DIR = BASE_DIR / "training_results"
TRAINING_REPORTS_DIR = BASE_DIR / "training_reports"
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"

DEFECT_TYPE_IDS = (2, 6, 10, 16, 19, 22, 23, 25, 27, 29, 30, 36)
DEFECT_TYPE_TO_YOLO_INDEX = {defect_type: idx for idx, defect_type in enumerate(DEFECT_TYPE_IDS)}
YOLO_INDEX_TO_DEFECT_TYPE = {idx: defect_type for defect_type, idx in DEFECT_TYPE_TO_YOLO_INDEX.items()}
YOLO_CLASS_NAMES = {idx: f"type_{defect_type:03d}" for idx, defect_type in YOLO_INDEX_TO_DEFECT_TYPE.items()}


def ensure_project_dirs() -> None:
    for directory in (TRAINING_RESULTS_DIR, TRAINING_REPORTS_DIR, TRAINED_MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def dataset_has_required_layout(dataset_dir: Path) -> bool:
    """Check whether a dataset directory has YOLO train/val/test image+label splits."""
    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"
    if not images_root.exists() or not labels_root.exists():
        return False

    for split in ("train", "val", "test"):
        if not (images_root / split).exists():
            return False
        if not (labels_root / split).exists():
            return False
    return True


def preferred_training_dataset_dir() -> Path:
    """
    Prefer tiled dataset first (which may already be overwritten by balanced data),
    then expanded/balanced snapshots if present.
    Fallback to the legacy dataset if tiled layout is unavailable.
    """
    if dataset_has_required_layout(REFINED_TILED_DATASET_DIR):
        return REFINED_TILED_DATASET_DIR
    if dataset_has_required_layout(TILED_DATASET_DIR):
        return TILED_DATASET_DIR
    if dataset_has_required_layout(EXPANDED_TILED_DATASET_DIR):
        return EXPANDED_TILED_DATASET_DIR
    if dataset_has_required_layout(BALANCED_TILED_DATASET_DIR):
        return BALANCED_TILED_DATASET_DIR
    return YOLO_DATASET_DIR


def dataset_yaml_text(dataset_dir: Path | None = None) -> str:
    dataset_dir = (dataset_dir or YOLO_DATASET_DIR).resolve()
    lines = [
        "# PF-MDD Dataset Configuration for YOLOv8",
        "# Plastic Film Multi-class Defect Detection (4096x256)",
        "",
        "# Dataset paths",
        f"path: {dataset_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "# Number of defect classes",
        f"nc: {len(YOLO_CLASS_NAMES)}",
        "",
        "# Class names (contiguous YOLO indices mapped from raw defect IDs)",
        "names:",
    ]
    for class_idx, class_name in YOLO_CLASS_NAMES.items():
        lines.append(f"  {class_idx}: {class_name}")
    lines.append("")
    return "\n".join(lines)


def _resolve_yaml_dataset_path(path_text: str, yaml_path: Path) -> Path:
    candidate = Path(path_text.strip())
    if not candidate.is_absolute():
        candidate = yaml_path.parent / candidate
    return candidate.resolve(strict=False)


def ensure_dataset_yaml(dataset_dir: Path | None = None, yaml_path: Path | None = None) -> Path:
    """Ensure pf_mdd.yaml points to the current workspace dataset directory."""
    dataset_dir = (dataset_dir or YOLO_DATASET_DIR).resolve()
    yaml_path = yaml_path or (dataset_dir / "pf_mdd.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    desired_text = dataset_yaml_text(dataset_dir)

    if yaml_path.exists():
        current_text = yaml_path.read_text(encoding="utf-8")
        if current_text.strip() == desired_text.strip():
            return yaml_path

        current_path = None
        for line in current_text.splitlines():
            if line.strip().startswith("path:"):
                current_path = line.split(":", 1)[1].strip()
                break

        if current_path:
            resolved_current = _resolve_yaml_dataset_path(current_path, yaml_path)
            if resolved_current == dataset_dir and f"nc: {len(YOLO_CLASS_NAMES)}" in current_text:
                return yaml_path

    yaml_path.write_text(desired_text + "\n", encoding="utf-8")
    return yaml_path


def resolve_data_path(path_like: str | Path) -> Path:
    """Resolve a possibly stale metadata path into the current workspace."""
    path = Path(path_like)
    if path.exists():
        return path

    parts = list(path.parts)
    if "data" in parts:
        data_index = parts.index("data")
        candidate = DATA_DIR.joinpath(*parts[data_index + 1 :])
        if candidate.exists():
            return candidate

    return path
