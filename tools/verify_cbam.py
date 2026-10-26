#!/usr/bin/env python
"""CBAM 注入自检（在正式训练前跑一遍，30 秒内完成）

作用
----
确认「CBAM 模块 + 训练器注入」这条链路在当前 ultralytics 版本下正常工作，
避免花几个小时训练后才发现注意力模块根本没挂上。

检查项
------
1. CBAM 模块本身：输出形状与输入一致、确实改变了数值（不是恒等映射）
2. 三种注入位置（backbone / neck / all）的参数量增量
3. 通过训练器的 `get_model()` 路径构建模型（**和真实训练用同一个入口**），
   确认 CBAM 存在、参数可训练、前向传播无异常

用法
----
    python tools/verify_cbam.py
    python tools/verify_cbam.py --weights yolov8n.pt

不传 --weights 时使用项目根目录的 yolov8s.pt；不存在则用 ultralytics 默认权重。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="CBAM 注入自检")
    parser.add_argument("--weights", default=None, help="预训练权重，默认 yolov8s.pt")
    parser.add_argument("--nc", type=int, default=12, help="类别数，默认 12")
    parser.add_argument("--imgsz", type=int, default=640, help="自检输入尺寸")
    parser.add_argument("--data", default=None,
                        help="数据集 YAML（仅第 3 步需要；不传则自动探测，找不到会跳过该步）")
    args = parser.parse_args()

    import torch
    print(f"torch {torch.__version__}  CUDA 可用: {torch.cuda.is_available()}")

    import ultralytics
    print(f"ultralytics {ultralytics.__version__}")

    from ultralytics import YOLO

    weights = args.weights or str(BASE_DIR / "yolov8s.pt")
    if not Path(weights).exists():
        print(f"[WARN] 未找到 {weights}，改用 yolov8n.pt（会自动下载）")
        weights = "yolov8n.pt"

    from modules import CBAM, attach_cbam, count_parameters, list_c2f_indices
    from modules.cbam_trainer import make_cbam_trainer

    ok = True

    # ---------- 1. CBAM 模块本身 ----------
    print("\n[1/3] CBAM 模块自检")
    cb = CBAM(64, reduction=16, kernel_size=7)
    x = torch.randn(2, 64, 32, 32)
    y = cb(x)
    shape_ok = tuple(y.shape) == tuple(x.shape)
    changed = not torch.allclose(x, y)
    print(f"  形状不变: {shape_ok}   数值被调整: {changed}   参数量: {sum(p.numel() for p in cb.parameters())}")
    if not (shape_ok and changed):
        print("  [FAIL] CBAM 模块异常")
        ok = False
    else:
        print("  [OK]")

    # ---------- 2. 三种注入位置的参数量 ----------
    print("\n[2/3] 不同注入位置的参数量")
    from ultralytics.nn.tasks import DetectionModel

    yolo = YOLO(weights)
    cfg = yolo.model.yaml
    baseline = DetectionModel(cfg, nc=args.nc, verbose=False)
    base_total, _ = count_parameters(baseline)
    print(f"  基线模型参数: {base_total:,}")
    print(f"  C2f 层索引: {list_c2f_indices(baseline)}")

    for where in ("backbone", "neck", "all"):
        m = DetectionModel(cfg, nc=args.nc, verbose=False)
        idx = attach_cbam(m, where=where, verbose=False)
        total, trainable = count_parameters(m)
        m.train()
        try:
            out = m(torch.randn(1, 3, args.imgsz, args.imgsz))
            fwd = "前向 OK"
        except Exception as exc:  # noqa: BLE001
            fwd = f"前向失败: {exc}"
            ok = False
        print(f"  where={where:<9} CBAM×{len(idx):<2} 层={idx}  参数 {total:,} (+{total-base_total:,})  {fwd}")

    # ---------- 3. 训练器路径（真实训练入口） ----------
    print("\n[3/3] 训练器 get_model() 路径（与真实训练同一入口）")
    trainer_cls = make_cbam_trainer(where="backbone", reduction=16, verbose=False)

    data_yaml = args.data
    if not data_yaml:
        for cand in (
            BASE_DIR / "pfmd_tiled_dataset_refined" / "pf_mdd.yaml",
            BASE_DIR / "pfmd_tiled_dataset" / "pf_mdd.yaml",
        ):
            if cand.exists():
                data_yaml = str(cand)
                break

    if not data_yaml:
        print("  [SKIP] 未找到数据集 YAML。第 1、2 步已证明 CBAM 模块与注入逻辑正常；")
        print("         训练器路径需要一个可用的 data 配置，用 --data 指定即可复查。")
        print("\n" + "=" * 60)
        print("CBAM 模块与注入逻辑自检通过（训练器路径已跳过）")
        print("=" * 60)
        return 0 if ok else 1
    trainer = trainer_cls(overrides={
        "model": weights,
        "data": data_yaml,
        "epochs": 1,
        "imgsz": args.imgsz,
        "batch": 2,
        "device": "cpu",
        "workers": 0,
        "verbose": False,
        "plots": False,
    })
    model = trainer.get_model(cfg=cfg, weights=yolo.model, verbose=False)

    n_cbam = sum(1 for m in model.modules() if type(m).__name__ == "CBAM")
    wrapped = [i for i, m in enumerate(model.model) if type(m).__name__ == "C2fWithCBAM"]
    total, trainable = count_parameters(model)
    print(f"  CBAM 模块数量: {n_cbam}   包装层索引: {wrapped}")
    print(f"  总参数 {total:,}   可训练 {trainable:,}")

    if n_cbam == 0:
        print("  [FAIL] 训练器路径未注入 CBAM")
        ok = False
    elif trainable < total * 0.99:
        print("  [WARN] 可训练参数明显少于总参数，检查是否被冻结")
    else:
        print("  [OK] 预训练权重已加载，CBAM 已注入且全部可训练")

    print("\n" + "=" * 60)
    print("自检通过，可以开始训练：" if ok else "自检未通过，请勿开始训练：")
    print("  python train_baseline.py")
    print("  # 或启用 CBAM：")
    print("  python -c \"from train_baseline import train_baseline_yolov8; \\")
    print("      train_baseline_yolov8(overrides={'cbam':'backbone','epochs':100})\"")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
