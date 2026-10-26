"""
PF-MDD 宽幅图切块策略单测

覆盖：切块数量、切块尺寸、全幅覆盖性、窄图退化、等比缩放填充、YOLO 框跨块坐标映射。

运行：pytest tests/test_image_processing.py -v
"""

import numpy as np
import pytest

from image_processing_strategy import WideImageProcessor


@pytest.fixture
def processor():
    """4096x256 线扫图像的标准处理器（640 切块，25% 重叠）"""
    return WideImageProcessor(original_size=(4096, 256), target_tile_size=640, overlap_ratio=0.25)


@pytest.fixture
def wide_image():
    """合成的 4096x256 BGR 图；左半亮、右半暗，便于检查切块是否覆盖全幅"""
    img = np.zeros((256, 4096, 3), dtype=np.uint8)
    img[:, :2048] = 200
    img[:, 2048:] = 50
    return img


# ---------- 切块数量与尺寸 ----------

def test_tile_count_matches_tiling_params(processor, wide_image):
    """切块数量必须等于参数计算得到的水平切块数，不多不少"""
    tiles = processor.tile_image_with_overlap(wide_image)
    assert len(tiles) == processor.n_tiles_horizontal


def test_every_tile_has_target_size(processor, wide_image):
    """每个切块都被 resize 成 target_size x target_size，否则 YOLO 输入维度不一致"""
    tiles = processor.tile_image_with_overlap(wide_image)
    assert all(tile.shape == (640, 640, 3) for tile, _ in tiles)


def test_tiles_cover_entire_width_without_gaps(processor, wide_image):
    """相邻切块必须重叠或首尾相接，不能出现未覆盖的列（漏检风险）"""
    tiles = processor.tile_image_with_overlap(wide_image)
    positions = [meta["position"] for _, meta in tiles]

    assert positions[0]["x_start"] == 0, "首个切块必须从第 0 列开始"
    assert positions[-1]["x_end"] == wide_image.shape[1], "末个切块必须覆盖到最后一列"

    for prev, cur in zip(positions, positions[1:]):
        assert cur["x_start"] <= prev["x_end"], (
            f"切块之间存在空隙：前块结束于 {prev['x_end']}，后块起始于 {cur['x_start']}"
        )


def test_narrow_image_degrades_to_single_tile():
    """宽度小于切块尺寸时退化为单块，不应产生 0 块或重复块"""
    proc = WideImageProcessor(original_size=(400, 256), target_tile_size=640)
    assert proc.n_tiles_horizontal == 1

    image = np.zeros((256, 400, 3), dtype=np.uint8)
    tiles = proc.tile_image_with_overlap(image)
    assert len(tiles) == 1
    assert tiles[0][0].shape == (640, 640, 3)


# ---------- 元数据 ----------

def test_metadata_records_scale_and_original_size(processor, wide_image):
    """元数据里的 scale / original_size 是坐标回映射的唯一依据，必须准确"""
    tiles = processor.tile_image_with_overlap(wide_image)
    _, meta = tiles[0]

    assert meta["original_size"] == (4096, 256)
    assert meta["tile_id"] == 0
    # y 方向：整幅高度 256 被拉到 640
    assert meta["scale"]["y"] == pytest.approx(640 / 256)


# ---------- 等比缩放 + 填充 ----------

def test_adaptive_resize_preserves_aspect_ratio_and_pads(processor, wide_image):
    """不切块时的替代方案：等比缩放 + YOLO 标准灰边填充（114）"""
    padded, meta = processor.adaptive_resize_and_pad(wide_image)

    assert padded.shape == (640, 640, 3)
    new_w, new_h = meta["resized_size"]
    assert new_w == 640 and new_h == 40, "4096x256 等比缩放到宽 640 后高应为 40"
    assert meta["padding"]["left"] == 0 and meta["padding"]["right"] == 0
    assert meta["padding"]["top"] + meta["padding"]["bottom"] == 640 - 40

    # 顶部填充区域应为 114（YOLO 官方灰）
    assert (padded[0, :, :] == 114).all()
    # 中间有效区域不应是纯填充色
    assert not (padded[320, :, :] == 114).all()


# ---------- YOLO 框跨块坐标映射 ----------

def test_box_inside_tile_is_remapped_into_tile_space(processor):
    """原图右端的框映射到最右切块后，x 坐标必须落在 (0,1] 区间内"""
    tiles = processor.tile_image_with_overlap(np.zeros((256, 4096, 3), dtype=np.uint8))
    last_meta = tiles[-1][1]

    # 框中心在 x=0.95*W 处，宽 0.02*W
    boxes = [[3, 0.95, 0.5, 0.02, 0.1]]
    converted = processor.convert_yolo_boxes_for_tile(boxes, last_meta)

    assert len(converted) == 1
    class_id, xc, yc, bw, bh = converted[0]
    assert class_id == 3
    assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
    assert 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0


def test_box_outside_tile_horizontal_range_is_dropped(processor):
    """完全落在切块水平范围外的框必须被丢弃，否则会污染该块标签"""
    tiles = processor.tile_image_with_overlap(np.zeros((256, 4096, 3), dtype=np.uint8))
    first_meta = tiles[0][1]

    # 框在 x=0.95*W，而首块只覆盖 x ∈ [0, 640/4096 ≈ 0.156]
    boxes = [[1, 0.95, 0.5, 0.01, 0.05]]
    assert processor.convert_yolo_boxes_for_tile(boxes, first_meta) == []


def test_conversion_is_deterministic(processor, wide_image):
    """同样的输入两次切块必须完全一致，保证数据集可复现"""
    a = processor.tile_image_with_overlap(wide_image)
    b = processor.tile_image_with_overlap(wide_image)
    assert len(a) == len(b)
    for (ta, ma), (tb, mb) in zip(a, b):
        assert np.array_equal(ta, tb)
        assert ma["position"] == mb["position"]
