# PF-MDD 项目文件与代码作用说明

本文档面向当前仓库 `E:\work\26032401\code1`，说明“文件做什么、代码做什么、如何串起来跑”。

## 1. 项目目标

本项目用于 **PF-MDD 塑料薄膜缺陷检测**，核心任务是把原始图像与掩码数据处理成 YOLOv8 可训练数据集，并完成训练、评估、实验追踪。

当前主训练数据集优先使用：`pfmd_tiled_dataset_refined`。

## 2. 当前目录与作用

| 路径 | 作用 | 是否核心 |
|---|---|---|
| `data/` | 原始数据与基础标注元数据（Defect/Mask/yolo_labels） | 是 |
| `pfmd_tiled_dataset/` | 从原始数据重建的基础 tiled 数据集 | 是 |
| `pfmd_tiled_dataset_refined/` | 重划分+增强后的主训练数据集 | 是 |
| `training_results/` | 训练运行输出（权重、曲线、指标） | 是 |
| `training_reports/` | 评估图表与报告 | 是 |
| `train_baseline.py` | YOLOv8 训练入口 | 是 |
| `evaluate_model.py` | 模型评估入口 | 是 |
| `verify_pipeline.py` | 数据与配置一致性检查入口 | 是 |
| `project_config.py` | 全局路径、类别映射、YAML 生成逻辑 | 是 |
| `step0~step10` | 数据处理流水线脚本 | 是（按流程选择） |

## 3. 配置中心：`project_config.py`

该文件是项目的“中枢配置”：

- 定义核心目录：`DATA_DIR`、`TILED_DATASET_DIR`、`REFINED_TILED_DATASET_DIR` 等。
- 定义类别映射：原始缺陷类型 ID ↔ YOLO 连续类 ID（0~11）。
- 提供数据集可用性判断：`dataset_has_required_layout()`。
- 自动生成/修复 YOLO YAML：`ensure_dataset_yaml()`。
- 训练集优先级选择：`preferred_training_dataset_dir()`（优先 refined）。

## 4. 代码主流程（推荐）

### 4.1 从原始数据重建基础训练集

1. `step0_rebuild_tiled_dataset_from_raw.py`
- 输入：`data/Defect_images/...` + `data/yolo_labels/*.txt`
- 处理：按原图分 train/val/test，然后进行宽图切片（tiling）并转换框到 tile 坐标
- 输出：`pfmd_tiled_dataset/`

### 4.2 构建 refined 数据集

2. `step8_refine_split_and_boost.py`
- 输入：`pfmd_tiled_dataset/`
- 处理：按 origin 组划分、清理跨 split 泄漏、train 少数类补到最小框数
- 输出：`pfmd_tiled_dataset_refined/` + `refine_summary.json`

### 4.3 扩充 train（模拟数据）

3. `step10_expand_refined_train_sim.py`
- 输入：`pfmd_tiled_dataset_refined/`
- 处理：仅扩充 train（翻转、旋转、亮度、gamma、噪声、模糊）
- 输出：扩充后的 `pfmd_tiled_dataset_refined/images/train` 与 `labels/train`，并写入 `step10_simulation_summary.json`

### 4.4 训练与评估

4. `verify_pipeline.py`
- 检查 YAML、标签范围、目录结构、统计文件。

5. `train_baseline.py`
- 使用 Ultralytics YOLO 进行训练。
- 自动选择当前可用且优先级最高的数据集（通常是 refined）。

6. `evaluate_model.py`
- 对指定 run 的 best.pt 做 test 评估，输出指标与图表。

7. `step9_experiment_tracker.py`
- 可选：自动训练+评估并把结果追加到 CSV（总览和分类别）。

## 5. 各脚本详细说明（step 系列）

### `step0_rebuild_tiled_dataset_from_raw.py`

- 类：`RawSample`、`TiledDatasetRebuilder`
- 作用：直接从 `data/` 重建新的 `pfmd_tiled_dataset`。
- 关键点：
- 按原始样本分组进行 split，避免源样本泄漏。
- 使用 `WideImageProcessor` 进行 4096x256 -> 640 tile 切分。
- 保存 `rebuild_summary.json`。

### `step1_integration.py`

- 类：`PFMDDBase`
- 作用：扫描/整合原始目录，生成 `pf_mdd_dataset_info.json` 元信息。
- 典型用途：数据入库和完整性检查。

### `step2_mask_to_yolo.py`

- 类：`MaskToYOLOConverter`
- 作用：从 mask 提取轮廓，转换成 YOLO 框标注。
- 输出：`data/yolo_labels/*.txt` 和 `conversion_stats.json`。

### `step3_data_augmentation.py`

- 类：`DataAugmentor`
- 作用：基于原始 defect 图做传统增强，输出到 `data/augmented_dataset/`。
- 当前状态：属于历史流程，可选。

### `step4_dataset_split.py`

- 类：`DatasetSplitter`
- 作用：把 `data/augmented_dataset` 划分为 train/val/test。
- 当前状态：属于历史流程，可选。

### `step5_imbalance_analysis.py`

- 类：`ImbalanceAnalyzer`
- 作用：分析类别不平衡程度并给出建议。

### `step6_expand_tiled_dataset.py`

- 类：`TiledDatasetExpander`
- 作用：仅扩充 train 正样本，生成 `pfmd_tiled_dataset_expanded/`。

### `step7_balance_tiled_dataset.py`

- 类：`TiledDatasetBalancer`
- 作用：按类别目标框数平衡 train，生成 `pfmd_tiled_dataset_balanced/`。

### `step8_refine_split_and_boost.py`

- 类：`Record`、`DatasetRefiner`
- 作用：构建更“可评估+可训练”的 refined 集。
- 关键点：
- origin 级划分。
- val/test 保持无增强污染。
- 去除跨 split 重复风险。
- train 少数类 boost。

### `step9_experiment_tracker.py`

- 作用：训练与评估结果写入 `training_results/experiment_tracking_*.csv`，便于多轮对比。

### `step10_expand_refined_train_sim.py`

- 类：`TrainSample`、`RefinedTrainSimulator`
- 作用：只针对 refined train 做大规模模拟扩充（你目前主用）。

## 6. 训练/评估代码说明

### `train_baseline.py`

- 核心函数：`train_baseline_yolov8()`
- 作用：
- 自动解析数据集 YAML。
- 校验标签类别范围。
- 配置并启动 YOLOv8 训练。
- 输出训练摘要到 `training_results/`。

### `evaluate_model.py`

- 类：`ModelEvaluator`
- 作用：
- 加载 best.pt。
- 在 test split 上评估。
- 导出总体与分类别指标、图表。

### `verify_pipeline.py`

- 作用：一键检查“能否训练”。
- 检查内容：元数据、转换统计、YAML、一致性、标签合法性。

## 7. 当前数据集状态（本次代码运行后的状态）

### `pfmd_tiled_dataset`

- train: 576 images / 86 labels / 490 backgrounds
- val: 120 images / 18 labels / 102 backgrounds
- test: 152 images / 21 labels / 131 backgrounds

### `pfmd_tiled_dataset_refined`

- train: 4451 images / 3312 labels / 1139 backgrounds
- val: 152 images / 28 labels / 124 backgrounds
- test: 145 images / 24 labels / 121 backgrounds

说明：
- 检测任务中，背景图没有标签文件是正常现象。
- 可训练前提是：无孤儿标签、标签范围合法、YAML 指向正确路径。

## 8. 推荐执行命令

```powershell
# 1) 从原始数据重建基础 tiled 数据集
python step0_rebuild_tiled_dataset_from_raw.py

# 2) 生成 refined 数据集
python step8_refine_split_and_boost.py

# 3) 对 refined train 做模拟扩充
python step10_expand_refined_train_sim.py --dataset-dir .\pfmd_tiled_dataset_refined --target-boxes-per-class 320 --max-new-images 7000 --background-ratio 0.25

# 4) 训练前检查
python verify_pipeline.py

# 5) 启动训练
python train_baseline.py

# 6) 评估
python evaluate_model.py
```

## 9. 文件保留建议

建议长期保留：
- `data/`
- `project_config.py`
- `step0_rebuild_tiled_dataset_from_raw.py`
- `step8_refine_split_and_boost.py`
- `step10_expand_refined_train_sim.py`
- `train_baseline.py`
- `evaluate_model.py`
- `verify_pipeline.py`
- `pfmd_tiled_dataset_refined/`（当前主训练集）

可按需归档：
- `training_results/`（历史实验大目录）
- `training_reports/`
- `pfmd_tiled_dataset/`（可由 step0 重新生成）
- `step1~step7`（若不再走旧流水线）

---

如果你后面愿意，我可以再给你补一版“新人接手指南”MD（只保留最短可执行流程 + 常见报错处理）。
