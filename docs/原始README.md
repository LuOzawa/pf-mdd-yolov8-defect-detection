> **⚠️ 本文档为原作者 README，已归档保留，仅供参考，内容与当前仓库不一致。**
>
> 已知差异（请以仓库根目录的 [README.md](../README.md) 为准）：
>
> | 原 README 的说法 | 当前实际情况 |
> | --- | --- |
> | `step1`~`step7` 位于项目根目录 | 已移入 `legacy_steps/` |
> | 最新训练结果在 `training_results/yolov8s_20260418_110331/` | 该目录**不存在**，`training_results/` 为空 |
> | 会生成 `training_reports/training_curves_comprehensive.png` 等 | 该目录为空，尚未产出评估图表 |
> | 无 `requirements.txt`，需手动装依赖 | 仓库已提供 `requirements.txt` |
> | `train_baseline.py` 固定 `device='cpu'` | 已改为 GPU 自动检测 |
> | 未提及 CBAM / 滑窗推理 | 已新增 `modules/`（CBAM）与 `sliding_window_inference.py` |
>
> 保留下来的价值：其中的数据处理流水线说明、参数含义与「已知现象与建议」一节仍有参考价值。

---

# PF-MDD YOLOv8 训练项目（当前版本）

本项目用于塑料薄膜缺陷检测（PF-MDD），基于 YOLOv8s 训练与评估，当前主数据集为 `pfmd_tiled_dataset`。

## 1. 当前项目状态

- 当前主训练集目录：`pfmd_tiled_dataset/`
- 类别数：`12`（`type_002` 到 `type_036` 的映射见 `project_config.py`）
- 当前主模型权重：`yolov8s.pt`
- 最新训练结果目录：`training_results/yolov8s_20260418_110331/`

当前数据集规模（`pfmd_tiled_dataset`）：

- train: 1931 images, 1487 labels, 444 backgrounds
- val: 192 images, 28 labels, 164 backgrounds
- test: 120 images, 21 labels, 99 backgrounds

## 2. 目录说明（按当前文件夹）

```text
code1/
├── data/                         # 原始与中间数据
│   ├── Defect_images/
│   ├── Mask_images/
│   ├── NODefect_images/
│   ├── yolo_labels/
│   └── pf_mdd_dataset_info.json
├── pfmd_tiled_dataset/           # 当前主训练集（YOLO 格式）
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   ├── pf_mdd.yaml
│   ├── expansion_summary.json
│   └── class_balance_summary.json
├── training_results/             # 训练输出
├── training_reports/             # 评估报告与图表
├── train_baseline.py             # 训练入口
├── evaluate_model.py             # 评估入口
├── verify_pipeline.py            # 项目/数据一致性检查
├── step1_integration.py          # 数据整合
├── step2_mask_to_yolo.py         # mask 转 YOLO 标签
├── step3_data_augmentation.py    # 传统增广（生成 data/augmented_dataset）
├── step4_dataset_split.py        # 划分到 pfmd_tiled_dataset
├── step5_imbalance_analysis.py   # 类不平衡分析
├── step6_expand_tiled_dataset.py # 扩增 tiled 数据集
├── step7_balance_tiled_dataset.py# 类别均衡扩增
├── step8_refine_split_and_boost.py # 重划分 + train少数类补到最小框数
├── step9_experiment_tracker.py  # 训练/评估指标自动追踪到CSV
└── project_config.py             # 全局路径与类别映射配置
```

## 3. 环境准备

建议 Python `3.8+`。无 `requirements.txt`，可手动安装核心依赖：

```powershell
python -m pip install ultralytics opencv-python numpy matplotlib pyyaml pandas
```

## 4. 快速开始（推荐）

### 4.1 检查数据与配置

```powershell
python verify_pipeline.py
```

说明：

- 该脚本会自动选择当前可用训练集（优先 `pfmd_tiled_dataset`）。
- 若 `split_statistics.json` 缺失，会显示 `[SKIP]`，不影响训练。

### 4.2 训练（默认配置）

```powershell
python train_baseline.py
```

默认训练关键参数（见 `train_baseline.py`）：

- epochs: 200
- batch: 8
- imgsz: 640
- device: cpu
- optimizer: AdamW

### 4.3 训练（常用调试参数）

```powershell
python -c "from train_baseline import train_baseline_yolov8; train_baseline_yolov8(overrides={'epochs':30,'workers':0})"
```

### 4.4 评估最新完整训练结果

```powershell
python evaluate_model.py
```

评估脚本会自动：

- 选取 `training_results/` 下最新且包含 `weights/best.pt` 的 run
- 自动解析该 run 对应的数据集 YAML
- 生成报告到 `training_reports/`

## 5. 数据处理流水线（从原始数据开始）

如果你需要从原始数据重新生成训练集，按顺序运行：

1. 数据整合

```powershell
python step1_integration.py
```

输出：`data/pf_mdd_dataset_info.json`

2. mask 转 YOLO 标签

```powershell
python step2_mask_to_yolo.py
```

输出：`data/yolo_labels/*.txt` 与 `data/yolo_labels/conversion_stats.json`

3. 基础增广

```powershell
python step3_data_augmentation.py
```

输出：`data/augmented_dataset/`

4. 划分训练/验证/测试集

```powershell
python step4_dataset_split.py
```

输出：`pfmd_tiled_dataset/`

## 6. 进阶数据扩增（可选）

### 6.1 扩增正样本 tiled 数据

```powershell
python step6_expand_tiled_dataset.py
```

输出：`pfmd_tiled_dataset_expanded/`

### 6.2 按类别做均衡扩增

```powershell
python step7_balance_tiled_dataset.py
```

输出：`pfmd_tiled_dataset_balanced/`

说明：

- 你当前项目中历史上已将均衡结果覆盖回 `pfmd_tiled_dataset`。
- 如果后续再次运行 step6/step7，会重新生成对应目录。

## 7. 数据集稳定化与对比追踪（本次新增）

### 7.1 一键完成“重划分 + train少数类补齐”

```powershell
python step8_refine_split_and_boost.py
```

输出：

- `pfmd_tiled_dataset_refined/`
- `pfmd_tiled_dataset_refined/refine_summary.json`

说明：该脚本会把 train 每个类别补到至少 50 框（若类别可生成），并提高 val/test 的正样本量稳定性。

### 7.2 持续训练并自动写入对比表

只评估并登记“最新完整 run”：

```powershell
python step9_experiment_tracker.py
```

训练后自动评估并登记（示例）：

```powershell
python step9_experiment_tracker.py --train --epochs 30 --workers 0
```

追踪文件：

- `training_results/experiment_tracking_summary.csv`
- `training_results/experiment_tracking_per_class.csv`
- `training_results/experiment_compare_report.md`

## 8. 关键配置说明

在 `project_config.py` 中：

- `YOLO_DATASET_DIR` 当前已对齐到 `pfmd_tiled_dataset`
- `preferred_training_dataset_dir()` 会优先选择可用 tiled 数据集
- `ensure_dataset_yaml()` 会自动修正 `pf_mdd.yaml` 的 `path` 到当前工作区

## 9. 常用命令速查

```powershell
# 1) 验证
python verify_pipeline.py

# 2) 训练（默认）
python train_baseline.py

# 3) 训练（30 epoch）
python -c "from train_baseline import train_baseline_yolov8; train_baseline_yolov8(overrides={'epochs':30,'workers':0})"

# 4) 评估
python evaluate_model.py

# 5) 类不平衡分析
python step5_imbalance_analysis.py

# 6) 重划分 + train少数类补齐
python step8_refine_split_and_boost.py

# 7) 自动追踪实验指标
python step9_experiment_tracker.py
```

## 10. 已知现象与建议

- 当前数据存在明显长尾与大量背景样本，`Recall` 可能偏低。
- 仅增加 epoch（如 200）不一定带来明显提升，数据分布与类别覆盖更关键。
- 先建议：
  - 用 `step5_imbalance_analysis.py` 看最新分布
  - 针对低频类别做定向增量
  - 每次训练后都执行 `evaluate_model.py` 对比 per-class 指标

## 11. 输出文件位置

- 训练 run：`training_results/<run_name>/`
- 训练摘要：`training_results/training_summary_*.json`
- 评估图表：
  - `training_reports/training_curves_comprehensive.png`
  - `training_reports/confusion_matrix.png`
- 评估报告：`training_reports/baseline_performance_report.md`

---

如果你准备继续做下一轮实验，建议流程：

1. `python verify_pipeline.py`
2. `python -c "from train_baseline import train_baseline_yolov8; train_baseline_yolov8(overrides={'epochs':30,'workers':0})"`
3. `python evaluate_model.py`
