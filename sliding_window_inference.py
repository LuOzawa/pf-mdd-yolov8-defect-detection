#!/usr/bin/env python
"""
宽幅图像滑窗推理 + 结果融合（对应任务书改进项四）

背景
----
PF-MDD 的原始图像是 4096x256 的宽幅图，而 YOLOv8 的输入是 640x640 方形。
训练阶段已经用切块（tiling）把宽图切成 640x640 训练，但**推理阶段**如果直接
把整张 4096x256 塞进模型，会被等比缩放到 640x40 左右，细小的缺陷几乎消失。

本脚本实现推理侧的完整流程：

    原图 → 重叠切块 → 逐块推理 → 坐标映射回原图 → NMS 融合 → 可视化

其中"坐标映射"复用 image_processing_strategy.WideImageProcessor.convert_yolo_boxes_for_tile
的逆运算，保证与训练时的切块规则一致。

使用方法
--------
    # 单图推理并保存可视化结果
    python sliding_window_inference.py --weights trained_models/best.pt \
        --source data/Defect_images/Defect_images/0001_002_00.png \
        --out runs/sliding_pred

    # 整个目录批量推理
    python sliding_window_inference.py --weights best.pt --source <图片目录> --out <输出目录>

    # 与"整图直接推理"做对比（论文里可直接作为消融实验）
    python sliding_window_inference.py --weights best.pt --source <图片目录> \
        --out runs/sw --compare-direct

参数说明
--------
--tile-size    切块边长，默认 640（与训练一致）
--overlap      相邻块重叠比例，默认 0.25
--conf         单块置信度阈值，默认 0.25
--iou          跨块融合时 NMS 的 IoU 阈值，默认 0.5
--out          输出目录，会写入 <输出目录>/predictions.json 与可视化图

注意
----
本脚本只依赖 ultralytics 的公开 API（YOLO.predict），不修改 ultralytics 内部实现，
因此不会因版本差异失效。运行前请确认已安装 requirements.txt 中的依赖。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    print("[ERROR] 缺少 opencv-python，请先执行: pip install -r requirements.txt")
    raise

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from project_config import YOLO_CLASS_NAMES  # noqa: E402


@dataclass
class Detection:
    """原图坐标系下的一个检测框。"""
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    tile_index: int = -1

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(self.x1), 2), round(float(self.y1), 2),
                     round(float(self.x2), 2), round(float(self.y2), 2)],
            "tile_index": self.tile_index,
        }


def compute_tiles(width: int, height: int, tile_size: int, overlap: float):
    """计算切块位置，返回 [(x, y, w, h), ...]，单位为原图像素。

    规则与训练时的切块保持一致：相邻块之间保留 tile_size * overlap 的重叠，
    保证跨切块边界的缺陷至少完整出现在一个块里。
    """
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    xs = list(range(0, max(1, width - tile_size + 1), step))
    ys = list(range(0, max(1, height - tile_size + 1), step))

    # 保证最右/最下边缘被覆盖
    if xs and xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if ys and ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))

    tiles = []
    for y in ys:
        for x in xs:
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            tiles.append((x, y, w, h))

    # 宽高不足一个切块时，退化为整图一次推理
    if not tiles:
        tiles = [(0, 0, width, height)]
    return tiles


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """按类别外做一次全局 NMS，用于合并跨切块的重复检测。

    boxes 形如 [[x1, y1, x2, y2], ...]（原图像素坐标）。
    返回保留下来的索引列表。
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


class SlidingWindowPredictor:
    """宽幅图像滑窗检测器。"""

    def __init__(self, weights: str | Path, tile_size: int = 640, overlap: float = 0.25,
                 conf: float = 0.25, iou: float = 0.5, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.tile_size = int(tile_size)
        self.overlap = float(overlap)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.class_names = YOLO_CLASS_NAMES

    def _predict_tile(self, tile: np.ndarray) -> list[Detection]:
        """对单个切块推理，返回切块坐标系下的检测。"""
        results = self.model.predict(
            source=tile,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            out.append(Detection(
                class_id=cls_id,
                class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))
        return out

    def predict_image(self, image: np.ndarray, merge_across_classes: bool = False):
        """滑窗推理一张宽幅图，返回 (融合后的检测列表, 切块数)。"""
        height, width = image.shape[:2]
        tiles = compute_tiles(width, height, self.tile_size, self.overlap)

        all_detections: list[Detection] = []
        for idx, (x, y, w, h) in enumerate(tiles):
            tile_img = image[y:y + h, x:x + w]
            if tile_img.size == 0:
                continue
            # 不足一个切块时用边缘复制填充，避免 ultralytics 内部缩放导致长宽比失真
            if w != self.tile_size or h != self.tile_size:
                tile_img = cv2.copyMakeBorder(
                    tile_img, 0, self.tile_size - h, 0, self.tile_size - w,
                    cv2.BORDER_REPLICATE,
                )
            for det in self._predict_tile(tile_img):
                # 坐标映射回原图
                det.x1 += x
                det.x2 += x
                det.y1 += y
                det.y2 += y
                # 裁掉落在填充区域内的框
                det.x1 = max(0.0, min(det.x1, width))
                det.x2 = max(0.0, min(det.x2, width))
                det.y1 = max(0.0, min(det.y1, height))
                det.y2 = max(0.0, min(det.y2, height))
                if det.x2 - det.x1 <= 1 or det.y2 - det.y1 <= 1:
                    continue
                det.tile_index = idx
                all_detections.append(det)

        merged = self._merge(all_detections, merge_across_classes)
        return merged, len(tiles)

    def _merge(self, detections: list[Detection], merge_across_classes: bool) -> list[Detection]:
        """跨切块融合。

        默认按类别分别做 NMS（同类才可能重复）；开启 merge_across_classes
        时把所有类别放在一起做 NMS，适合不同类别框高度重叠的场景。
        """
        if not detections:
            return []

        grouped: dict[int | None, list[Detection]] = {}
        for det in detections:
            key = None if merge_across_classes else det.class_id
            grouped.setdefault(key, []).append(det)

        merged: list[Detection] = []
        for _, dets in grouped.items():
            boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
            scores = np.array([d.confidence for d in dets], dtype=np.float32)
            keep = nms_numpy(boxes, scores, self.iou)
            merged.extend(dets[i] for i in keep)

        merged.sort(key=lambda d: d.confidence, reverse=True)
        return merged

    def predict_direct(self, image: np.ndarray) -> list[Detection]:
        """对照组：整图直接推理（不复用切块）。

        用于论文里证明"滑窗融合相对整图推理的收益"。
        """
        results = self.model.predict(source=image, conf=self.conf, iou=self.iou,
                                     verbose=False, device=self.device)
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None:
            return out
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            out.append(Detection(
                class_id=cls_id,
                class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                confidence=float(box.conf[0]),
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))
        return out


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """在原图上画框，返回可视化图像。"""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for det in detections:
        color = (0, 200, 0)
        cv2.rectangle(vis, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (int(det.x1), int(det.y1) - th - 4),
                      (int(det.x1) + tw + 4, int(det.y1)), color, -1)
        cv2.putText(vis, label, (int(det.x1) + 2, int(det.y1) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def iter_images(source: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if source.is_file():
        yield source
        return
    for p in sorted(source.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(description="PF-MDD 宽幅图像滑窗推理 + 结果融合")
    parser.add_argument("--weights", required=True, help="训练好的权重路径，如 trained_models/best.pt")
    parser.add_argument("--source", required=True, help="单张图片或图片目录")
    parser.add_argument("--out", default="runs/sliding_predict", help="输出目录")
    parser.add_argument("--tile-size", type=int, default=640, help="切块边长，默认 640")
    parser.add_argument("--overlap", type=float, default=0.25, help="重叠比例，默认 0.25")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="融合 NMS 的 IoU 阈值")
    parser.add_argument("--device", default=None, help="推理设备，如 0 或 cpu；默认自动")
    parser.add_argument("--compare-direct", action="store_true",
                        help="额外跑一遍整图直接推理作为对照，用于消融实验")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"[ERROR] 输入不存在: {source}")
        return 2

    out_dir = Path(args.out)
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    from train_baseline import pick_device
    device = pick_device(args.device)

    predictor = SlidingWindowPredictor(
        weights=args.weights,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf=args.conf,
        iou=args.iou,
        device=device,
    )

    summary = []
    for img_path in iter_images(source):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] 读取失败，跳过: {img_path}")
            continue

        detections, n_tiles = predictor.predict_image(image)
        record = {
            "image": img_path.name,
            "size": [int(image.shape[1]), int(image.shape[0])],
            "tiles": n_tiles,
            "num_detections": len(detections),
            "detections": [d.to_dict() for d in detections],
        }

        if args.compare_direct:
            direct = predictor.predict_direct(image)
            record["direct_num_detections"] = len(direct)
            record["direct_detections"] = [d.to_dict() for d in direct]

        summary.append(record)

        vis = draw_detections(image, detections)
        cv2.imwrite(str(out_dir / "vis" / img_path.name), vis)
        print(f"[OK] {img_path.name}: {len(detections)} 个检测（{n_tiles} 个切块）"
              + (f"，整图直推 {record['direct_num_detections']} 个" if args.compare_direct else ""))

    (out_dir / "predictions.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    total_sw = sum(r["num_detections"] for r in summary)
    print(f"\n完成：{len(summary)} 张图，滑窗共 {total_sw} 个检测 -> {out_dir}")
    if args.compare_direct:
        total_direct = sum(r.get("direct_num_detections", 0) for r in summary)
        print(f"对照：整图直推共 {total_direct} 个检测")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
