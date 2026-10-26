"""带 CBAM 的 YOLOv8 训练器

为什么需要这个文件
------------------
ultralytics 在 `model.train()` 时**不会沿用 `YOLO(...).model` 这个对象**，
而是由 `DetectionTrainer.get_model()` 依据 `model.yaml` **重新构建**一个
`DetectionModel`，再把预训练权重 load 进去。

所以「对 `YOLO(W).model` 注入 CBAM」在训练时会被丢弃。正确做法是重写
`get_model()`，在模型构建完成、权重加载完成之后立即注入 CBAM：

    构建 DetectionModel(cfg, nc=12)
        ↓
    load(预训练权重)            ← C2f 等主干参数来自 COCO 预训练
        ↓
    attach_cbam(...)           ← CBAM 随机初始化，参与训练

这正好符合「预训练主干 + 新增注意力模块」的常见做法。

用法
----
    from modules.cbam_trainer import make_cbam_trainer
    trainer_cls = make_cbam_trainer(where="backbone", reduction=16)
    model.train(trainer=trainer_cls, **train_config)

注意：CBAM 的配置走**类属性**而不是 `overrides`。
因为 ultralytics 的 `get_cfg()` 会拒绝未知的 override 键，传 `cbam_where=...`
会直接报错。
"""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer

from .attach_cbam import attach_cbam


class CBAMDetectionTrainer(DetectionTrainer):
    """在模型构建完成后自动注入 CBAM 的检测训练器。"""

    # 由 make_cbam_trainer() 覆盖
    cbam_where: str | list[int] = "backbone"
    cbam_reduction: int = 16
    cbam_kernel_size: int = 7
    cbam_verbose: bool = True

    def get_model(self, cfg=None, weights=None, verbose=True):
        # 父类内部：DetectionModel(cfg, nc=..., ch=...) + model.load(weights)
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        attached = attach_cbam(
            model,
            where=self.cbam_where,
            reduction=self.cbam_reduction,
            kernel_size=self.cbam_kernel_size,
            verbose=self.cbam_verbose,
        )
        if not attached:
            raise RuntimeError("CBAM 注入失败：未匹配到任何 C2f 层")
        return model


def make_cbam_trainer(where: str | list[int] = "backbone", reduction: int = 16,
                      kernel_size: int = 7, verbose: bool = True) -> type[CBAMDetectionTrainer]:
    """生成一个带指定 CBAM 配置的训练器类。

    用工厂函数而不是往 overrides 里塞参数，是为了绕开 ultralytics 的
    ``get_cfg()`` 未知键校验。
    """
    class _ConfiguredCBAMTrainer(CBAMDetectionTrainer):
        pass

    _ConfiguredCBAMTrainer.cbam_where = where
    _ConfiguredCBAMTrainer.cbam_reduction = reduction
    _ConfiguredCBAMTrainer.cbam_kernel_size = kernel_size
    _ConfiguredCBAMTrainer.cbam_verbose = verbose
    _ConfiguredCBAMTrainer.__name__ = f"CBAMDetectionTrainer_{where}"
    return _ConfiguredCBAMTrainer
