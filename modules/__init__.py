"""PF-MDD 自定义模型模块。

包含用于 YOLOv8 改进的注意力模块与注入工具：

- ``cbam``：CBAM 注意力模块（通道 + 空间）
- ``attach_cbam``：把 CBAM 挂到已加载模型的 C2f 模块之后
"""

from .cbam import CBAM, ChannelAttention, SpatialAttention
from .attach_cbam import attach_cbam, count_parameters, list_c2f_indices

__all__ = [
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "attach_cbam",
    "count_parameters",
    "list_c2f_indices",
]
