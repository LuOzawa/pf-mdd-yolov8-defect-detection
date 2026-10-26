"""把 CBAM 注入 YOLOv8（不改 ultralytics 源码、不改模型 YAML）

设计说明
--------
ultralytics 的模型由 `parse_model()` 依据 YAML 构建，若要在 YAML 里新增自定义
模块，需要 monkeypatch `parse_model`，这会随 ultralytics 版本失效，风险高。

这里改用**加载后注入**的方式：模型构建完成后再把每个 `C2f` 模块包一层
`C2fWithCBAM`（原模块 → CBAM）。好处：

1. 完全不碰 ultralytics 内部实现，升级不失效；
2. 可以用任意预训练权重（yolov8n/s/m/l/x）初始化；
3. 消融实验只需改一个参数（挂在哪几层），不用维护多份 YAML。

包装器会透传 ultralytics 需要的元信息（``i`` / ``f`` / ``type`` / ``np``），
因此前向传播的跨层引用（Concat 按索引取特征）不受影响。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cbam import CBAM


class C2fWithCBAM(nn.Module):
    """在原有模块输出后接一个 CBAM，保持张量形状不变。"""

    def __init__(self, base: nn.Module, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.base = base
        self.cbam = CBAM(channels, reduction=reduction, kernel_size=kernel_size)
        # 透传 ultralytics 在 _predict_once 中会用到的属性
        for attr in ("i", "f", "type", "np"):
            if hasattr(base, attr):
                setattr(self, attr, getattr(base, attr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(self.base(x))


def list_c2f_indices(model) -> list[int]:
    """返回模型中所有 C2f 模块的层索引。"""
    inner = _inner_layers(model)
    return [i for i, m in enumerate(inner) if type(m).__name__ == "C2f"]


def _inner_layers(model):
    """取出 ultralytics 模型的 nn.Sequential 主干。"""
    net = getattr(model, "model", model)
    inner = getattr(net, "model", net)
    return inner


def _infer_out_channels(module: nn.Module) -> int | None:
    """推断模块输出通道数。

    不同 ultralytics 版本内部命名可能不同，按优先级依次尝试：
    ``cv2.conv`` → ``cv3.conv`` → ``conv``。
    """
    for path in (("cv2", "conv"), ("cv3", "conv"), ("conv",)):
        obj: object = module
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "out_channels"):
            return int(obj.out_channels)
    return None


def attach_cbam(model, where: str = "backbone", reduction: int = 16,
                kernel_size: int = 7, verbose: bool = True) -> list[int]:
    """给模型中的 C2f 模块注入 CBAM。

    参数
    ----
    model : ultralytics.nn.tasks.DetectionModel 或 YOLO(...).model
    where : ``"backbone"`` / ``"neck"`` / ``"all"``，或显式层索引列表
        YOLOv8 的 C2f 分布：前 4 个在 backbone，后 4 个在 neck。
    reduction : 通道注意力的压缩比，默认 16
    kernel_size : 空间注意力卷积核，必须是奇数，默认 7
    verbose : 是否打印注入详情

    返回
    ----
    实际注入的层索引列表
    """
    inner = _inner_layers(model)
    c2f_indices = list_c2f_indices(model)
    if not c2f_indices:
        raise RuntimeError("模型中未找到 C2f 模块，请确认是 YOLOv8 系列模型")

    if isinstance(where, (list, tuple, set)):
        selected = [i for i in c2f_indices if i in set(where)]
    elif where == "all":
        selected = list(c2f_indices)
    elif where == "backbone":
        selected = c2f_indices[: len(c2f_indices) // 2]
    elif where == "neck":
        selected = c2f_indices[len(c2f_indices) // 2:]
    else:
        raise ValueError(f"where 参数非法: {where}（可选 backbone/neck/all 或索引列表）")

    attached: list[int] = []
    for idx in selected:
        base = inner[idx]
        channels = _infer_out_channels(base)
        if channels is None:
            if verbose:
                print(f"[WARN] 层 {idx} 无法推断输出通道，跳过")
            continue
        inner[idx] = C2fWithCBAM(base, channels, reduction=reduction, kernel_size=kernel_size)
        attached.append(idx)
        if verbose:
            print(f"[OK] 层 {idx:>2}  接 CBAM  channels={channels}")

    if verbose:
        print(f"[OK] 共注入 {len(attached)} 个 CBAM（where={where}, reduction={reduction}）")
    return attached


def count_parameters(model) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
