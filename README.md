# PF-MDD：基于 YOLOv8 的塑料薄膜多类表面缺陷检测

[![CI](https://github.com/LuOzawa/pf-mdd-yolov8-defect-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/LuOzawa/pf-mdd-yolov8-defect-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


面向挤出吹膜产线的塑料薄膜表面缺陷自动检测研究。以 **YOLOv8** 为核心框架，
针对 4096×256 宽幅细长图像设计**切块（tiling）+ 重叠 + 坐标回映射**的处理方案，
目标是在 12 类薄膜表面缺陷上实现可落地的自动检测。

- **数据集**：PF-MDD（Plastic Film Multi-class Defect Detection）
- **缺陷类别**：12 类（原始编号 002 / 006 / 010 / 016 / 019 / 022 / 023 / 025 / 027 / 029 / 030 / 036）
- **原始图像**：4096×256 单通道工业线扫图
- **已实现的改进**：CBAM 注意力（`modules/`）、宽幅图滑窗推理融合（`sliding_window_inference.py`）

项目为本科毕业设计作品。

---

## ⚠️ 先读这一节：当前完成度

本仓库是**数据工程与训练/评估框架**，模型训练与实验结果**尚未产出**。
下表是逐项对照任务书的实际状态，请以此为准，不要被代码量误导。

| 任务书要求 | 状态 | 说明 |
| --- | --- | --- |
| 4.1 数据集构建与预处理 | ✅ 完成 | 整合、掩码转 YOLO、切块、分层划分、类别均衡全部实现并跑通 |
| 4.2 YOLOv8 基线模型构建 | 🟡 代码就绪，**未训练** | `train_baseline.py` 可用，但仓库内**没有训练好的权重** |
| 4.2 基线性能评估 | ❌ 未产出 | `training_results/`、`training_reports/` 为空，无 mAP / FPS 数据 |
| 4.3 改进（至少 2 项） | ✅ 代码完成 2 项（待训练验证） | **改进一 CBAM 注意力** + **改进四 滑窗推理融合**，均已实现并通过注入自检；FPN 优化 / Focal Loss / 蒸馏剪枝未做 |
| 4.4 横向对比实验 | ❌ 未做 | 未与 YOLOv5s / YOLOv7 / Faster R-CNN / RT-DETR 对比 |
| 4.5 论文撰写 | — | 不在本仓库范围 |
| 技术指标 mAP@0.5 ≥ 85% | ❌ 无法验证 | 无训练产物 |
| 技术指标 FPS ≥ 30 | ❌ 无法验证 | 无训练产物；当前开发机无 CUDA |
| 技术指标 12 类全覆盖 | ⚠️ 受数据限制 | 测试集中 `type_027` / `type_029` / `type_036` **零样本** |

**结论**：这是一份「有完整流水线、两项改进已落地、但缺实验结论」的工程基础。
若要用于答辩，必须完成**训练并产出指标**，并补齐 **4.4 横向对比**；
4.3 的两项改进代码已完成，但「提升了多少」仍需实验数据支撑。

---

## 一、数据集现状（重要）

### 1.1 原始数据规模

| 项 | 数量 | 说明 |
| --- | --- | --- |
| 缺陷原图 | 106 张 | 4096×256 |
| 掩码标注 | 107 张 | 像素级二值掩码 |
| 正常样本（负样本） | 7 张 | 无缺陷 |
| 转换成功的 YOLO 标注 | 104 个 txt | 共 **138 个边界框** |

原始 12 类的真实框分布**极度不均衡**：

| 类别 | 真实框数 | 类别 | 真实框数 |
| --- | --- | --- | --- |
| `type_002` | 10 | `type_025` | 5 |
| `type_006` | 32 | `type_027` | 2 |
| `type_010` | 9 | `type_029` | **1** |
| `type_016` | 4 | `type_030` | 19 |
| `type_019` | 41 | `type_036` | **1** |
| `type_022` | 9 | `type_023` | 5 |

最多类（`type_019`，41 框）与最少类（`type_029` / `type_036`，各 1 框）相差 **41 倍**。

### 1.2 三个数据集版本的关系

```
原始 106 张 4096×256 图 + 掩码
   │
   ├─ step0 / step4 → pfmd_tiled_dataset/          切块后基础版
   │                    train 576 图 / 86 标注 / 110 框
   │                    val   120 图 / 18 标注 /  24 框
   │                    test  152 图 / 21 标注 /  25 框
   │
   ├─ step8 → 重划分 + train 少数类补到 ≥50 框
   │
   └─ step10 → train 每类补到 320 框（几何/光度增广）
                 → pfmd_tiled_dataset_refined/   ← 当前主训练集
                    train 4451 图 / 3312 标注 / 3840 框（每类恰好 320）
                    val    152 图 /   28 标注 /   47 框
                    test   145 图 /   24 标注 /   33 框
```

### 1.3 必须知道的三个事实

**① 训练集有 78% 是增广副本**

4451 张训练图中 3492 张（78%）由 `step10` 通过水平/垂直翻转、旋转、亮度、
Gamma、高斯噪声、模糊等变换生成。真实的（切块）训练图只有约 959 张。
每类恰好 320 框是为了类别均衡**人为造出来的**，不是真实分布。

**② 测试集只有 33 个框，且 3 个类别零样本**

| 类别 | test 框数 |
| --- | --- |
| `type_002` | 3 |
| `type_006` | 15 |
| `type_010` | 2 |
| `type_016` | 1 |
| `type_019` | 7 |
| `type_022` | 1 |
| `type_023` | 2 |
| `type_025` | 1 |
| `type_027` | **0** |
| `type_029` | **0** |
| `type_030` | 1 |
| `type_036` | **0** |

在 33 个框上计算全类 mAP，方差极大，**不能作为「12 类全覆盖」的证据**。
`evaluate_model.py` 因此专门提供了 `stable_macro_*` 指标（只统计样本数达标的类别），
论文中建议以此为主指标，并在正文中明确说明测试集的规模限制。

**③ 增广只作用于 train，val/test 保持真实**

这点方法学上是正确的（`step10` 注释写明 "Keep val/test unchanged for fair evaluation"，
实测 val/test 无任何增广文件名标记）。所以**测试集结论是干净的**，问题只在于样本太少。

---

## 二、代码结构

```
.
├── project_config.py                    # 全局配置：路径、类别映射、YAML 生成与自动修复
├── image_processing_strategy.py         # 宽幅图切块核心（512 行）
│                                        #   WideImageProcessor：重叠切块 / 坐标回映射 /
│                                        #   自适应缩放填充 / 切块可视化
├── train_baseline.py                    # YOLOv8 训练入口（GPU 自动检测）
├── evaluate_model.py                    # 评估入口（611 行，含类别稳定性分析）
├── verify_pipeline.py                   # 数据与配置一致性自检（纯只读）
├── sliding_window_inference.py          # ★ 滑窗推理 + 跨块 NMS 融合（改进四，383 行）
│
├── modules/                             # ★ 自定义模型模块
│   ├── cbam.py                          #   CBAM 注意力（通道 + 空间）
│   ├── attach_cbam.py                   #   加载后注入工具（不碰 ultralytics 源码）
│   └── cbam_trainer.py                  #   训练器子类（在 get_model 里注入）
├── tools/
│   └── verify_cbam.py                   #   CBAM 注入自检（训练前 30 秒跑一遍）
│
├── step0_rebuild_tiled_dataset_from_raw.py   # 从原始数据重建切块数据集
├── step8_refine_split_and_boost.py           # 重划分 + train 少数类补框
├── step9_experiment_tracker.py               # 实验指标自动追踪 → CSV
├── step10_expand_refined_train_sim.py        # train 增广到每类 320 框
│
├── legacy_steps/                        # 早期流水线（step1~step7）
│   ├── step1_integration.py             #   三个子数据集整合
│   ├── step2_mask_to_yolo.py            #   掩码 → YOLO 标注
│   ├── step3_data_augmentation.py       #   传统增广
│   ├── step4_dataset_split.py           #   7:2:1 分层划分
│   ├── step5_imbalance_analysis.py      #   类别不平衡分析
│   ├── step6_expand_tiled_dataset.py    #   切块数据扩增
│   └── step7_balance_tiled_dataset.py   #   类别均衡扩增
│
├── datasets/                            # 数据集「元数据」（不含图片）
│   ├── raw_metadata/
│   │   ├── pf_mdd_dataset_info.json     #   原始数据集清单
│   │   ├── conversion_stats.json        #   掩码转换统计
│   │   └── yolo_labels/*.txt            #   104 个标注文件（138 框，复现必备）
│   ├── pfmd_tiled_dataset/
│   │   ├── pf_mdd.yaml  rebuild_summary.json
│   ├── pfmd_tiled_dataset_refined/
│       ├── pf_mdd.yaml  refine_summary.json  step10_simulation_summary.json
│
├── docs/
│   ├── PROJECT_FILE_CODE_GUIDE.md       # 原项目文件说明
│   ├── PF_MDD_Imbalance_Analysis_Report.txt  # 类别不平衡分析与改进建议
│   ├── 原始README.md                     # 原作者 README（部分内容已过时，见下）
│   ├── tiling_visualization.png         # 切块策略可视化
│   ├── 数据集说明.md                           # 任务书要求的交付物
│   └── 启动说明.md                             # 环境配置与运行步骤
│
├── requirements.txt                     # 依赖清单
├── .gitignore                           # 排除数据图片 / 权重 / 训练输出
└── .gitattributes                       # 行尾与二进制类型约定
```

> **关于根目录的 `step*.py`**：这些是数据/训练流水线的顺序步骤脚本（`step0` → `step10`），
> 全部 `import project_config`（根目录，集中提供路径与超参）。
> 保持同层是为了让 `python step8_refine_split_and_boost.py` 无需设置 `PYTHONPATH` 即可直接运行——
> 这是刻意的工程取舍，不是目录规划遗漏。

**代码规模**（`wc -l` 实测）：

| 分组 | 文件数 | 行数 |
| --- | --- | --- |
| 主流程脚本（根目录 `*.py`） | 10 | 3,775 |
| `modules/`（CBAM 与注入） | 4 | 312 |
| `tools/`（自检） | 1 | 158 |
| `legacy_steps/`（早期流水线） | 7 | 2,191 |
| **合计** | **22** | **6,436** |

---

## 三、快速开始

> 详细步骤见 **[docs/启动说明.md](docs/启动说明.md)**。

```bash
# 1. 安装依赖
pip install -r requirements.txt
# GPU 用户请先按 CUDA 版本安装 PyTorch：https://pytorch.org/get-started/locally/

# 2. 准备数据（仓库未包含图片，见「仓库说明」）
#    把原始 Defect_images / Mask_images / NODefect_images 放到 data/ 下，
#    或直接复用已有的 pfmd_tiled_dataset_refined/ 目录

# 3. 自检
python verify_pipeline.py

# 4. 训练（GPU 自动检测，可用 YOLO_DEVICE 环境变量覆盖）
python train_baseline.py
# 或小配置快速验证链路
python -c "from train_baseline import train_baseline_yolov8; \
           train_baseline_yolov8(overrides={'epochs':30,'workers':0,'imgsz':640})"

# 5. 评估
python evaluate_model.py

# 6. 滑窗推理（改进四）
python sliding_window_inference.py \
    --weights training_results/<run>/weights/best.pt \
    --source data/Defect_images/Defect_images \
    --out runs/sliding_predict --compare-direct
```

### 关于 `SER/dev` 设备选择

`train_baseline.py` 的 `pick_device()` 按以下顺序决定训练设备：

1. `overrides={'device': '0'}` 显式指定
2. 环境变量 `YOLO_DEVICE=0`
3. 自动检测：有 CUDA 用 `"0"`，否则退回 `"cpu"`

> 原代码把 `device` 写死为 `'cpu'`，在有 GPU 的机器上会慢几十倍，也让 FPS 指标无从测量。

---

## 四、改进项 A：滑窗推理 + 结果融合（对应任务书改进四）

`sliding_window_inference.py` 实现了推理侧的完整链路：

```
原图(4096×256) → 重叠切块(640×640) → 逐块推理 → 坐标映射回原图 → 按类 NMS 融合 → 可视化
```

| 特性 | 说明 |
| --- | --- |
| 切块策略 | 与训练一致：`tile_size=640`、`overlap=0.25`，自动补齐右/下边缘 |
| 边缘处理 | 不足一格的块用 `BORDER_REPLICATE` 填充，避免长宽比失真 |
| 坐标映射 | 检出框平移回原图坐标，并裁剪掉落在填充区的无效框 |
| 结果融合 | 按类别分别做 NMS（可选跨类合并），保留置信度最高者 |
| 对照实验 | `--compare-direct` 同时跑「整图直接推理」，可直接作为消融数据 |

**实测（纯函数层，不依赖模型）**：

- 4096×256 → 9 个切块，右边界覆盖到 4096、下边界覆盖到 256 ✅
- 320×240（小于切块尺寸）→ 退化为 1 个整图块 ✅
- NMS：两个 IoU≈0.9 的重复框 + 一个远处框 → 输出 `[0, 2]`，正确抑制重复 ✅

> ⚠️ 该脚本的**模型推理部分尚未实际运行验证**（本机无 CUDA 且未安装 torch/ultralytics）。
> 纯逻辑（切块、坐标映射、NMS）已单测通过，接口调用使用 ultralytics 公开 API，
> 首次运行若报错请优先检查 ultralytics 版本。

---

## 五、改进项 B：CBAM 注意力（对应任务书改进一）

`modules/cbam.py` 实现与原论文一致的 CBAM：通道注意力（avg/max 双路 MLP 共享）
→ 空间注意力（通道维 avg/max 拼接 + 7×7 卷积）。

### 为何不能直接注入 `YOLO(...).model`

排查时发现两个坑（已绕开）：

1. `YOLO('.pt').model` 的**所有参数 `requires_grad=False`**
2. ultralytics 在 `model.train()` 时**不会沿用这个对象**，而是由
   `DetectionTrainer.get_model()` 依据 `model.yaml` **重新构建**模型再把权重 load 进去

所以对 `YOLO(...).model` 注入 CBAM 在训练时会被直接丢弃。正确做法是重写
`get_model()`：

```
构建 DetectionModel(cfg, nc=12)
    ↓
load(预训练权重)          ← C2f 等主干参数来自 COCO
    ↓
attach_cbam(...)         ← CBAM 随机初始化，参与训练
```

### 实现方式

| 文件 | 作用 |
| --- | --- |
| `modules/cbam.py` | CBAM 模块本体，不依赖 ultralytics |
| `modules/attach_cbam.py` | 把一个模块包成 `XxxWithCBAM`，保持张量形状不变，并透传 `i/f/type` 等元信息 |
| `modules/cbam_trainer.py` | `CBAMDetectionTrainer`：在 `get_model()` 中注入 |
| `tools/verify_cbam.py` | 自检脚本 |

**不采用 YAML + monkeypatch `parse_model` 的原因**：那需要改动 ultralytics 内部实现，
且解析器不会给未注册模块自动补 `c1`，换版本或改 width 就容易失效。

### 三种注入位置（实测参数量）

| 位置 | 层索引 | CBAM 个数 | 参数量 | 增量 |
| --- | --- | --- | --- | --- |
| `backbone`（推荐） | 2, 4, 6, 8 | 4 | 11,184,156 | +43,912 |
| `neck` | 12, 15, 18, 21 | 4 | 11,191,836 | +51,592 |
| `all` | 全部 8 个 | 8 | 11,235,748 | +95,504 |

基线 YOLOv8s（nc=12）为 11,140,244 参数。

### 启用方式

```bash
# 方式一：改 overrides
python -c "from train_baseline import train_baseline_yolov8; \
    train_baseline_yolov8(overrides={'cbam':'backbone','epochs':100})"

# 方式二：先自检，再训练
python tools/verify_cbam.py --data pfmd_tiled_dataset_refined/pf_mdd.yaml
python train_baseline.py
```

可用参数：`cbam` = `backbone` / `neck` / `all`，`cbam_reduction` = 通道压缩比（默认 16）。

### 实测验证结果

```
[1/3] CBAM 模块自检
  形状不变: True   数值被调整: True   参数量: 610

[2/3] 不同注入位置的参数量
  基线模型参数: 11,140,244
  C2f 层索引: [2, 4, 6, 8, 12, 15, 18, 21]
  where=backbone  CBAM×4  参数 11,184,156 (+43,912)  前向 OK
  where=neck      CBAM×4  参数 11,191,836 (+51,592)  前向 OK
  where=all       CBAM×8  参数 11,235,748 (+95,504)  前向 OK

[3/3] 训练器 get_model() 路径
  Transferred 349/355 items from pretrained weights
  CBAM 模块数量: 4   包装层索引: [2, 4, 6, 8]
  总参数 11,184,156   可训练 11,184,140
  [OK] 预训练权重已加载，CBAM 已注入且全部可训练
```

另做过一次不训练的 dry-run，确认 `cbam` 参数走 trainer 通道、**不会泄漏**
进 ultralytics 的配置（否则 `get_cfg()` 会直接报未知键）。

> 验证环境：torch 2.4.1+cpu、ultralytics 8.4.189、Python 3.8、YOLOv8s。
> **未做**：完整训练的收敛性与精度提升 —— 需在 GPU 机器上跑对照实验才能给出。

---

## 六、已知问题与风险

| 级别 | 问题 | 影响 |
| --- | --- | --- |
| 🔴 | 无训练产物（权重 / results.csv / 报告全空） | 无法证明任何技术指标 |
| 🟡 | 两项改进**已实现但未训练验证** | 代码与注入逻辑已自检，但「CBAM 带来多少 mAP 提升」需 GPU 对照实验才能给出 |
| 🔴 | 任务书 4.4 横向对比完全缺失 | 缺少对比实验章节素材 |
| 🟠 | 测试集仅 33 框，3 类零样本 | 全类 mAP 统计不可靠 |
| 🟠 | 训练集每类恰好 320 框（人为造平），78% 为增广副本 | 结论需说明「学到的是缺陷还是增广」 |
| 🟡 | `docs/原始README.md` 描述已过时 | 其中 `step1~step7` 现在位于 `legacy_steps/`；提到的 `training_results/yolov8s_20260418_110331/` 实际不存在 |
| 🟡 | 数据集 YAML 中的 `path` 为旧绝对路径 | 靠 `ensure_dataset_yaml()` 运行时自动修复 |
| ⚪ | 无 GPU 时训练不可行 | 4451 图 × 640 × 200 epoch on CPU 需数天以上 |

### 关于「Focal Loss」的更正

原 `train_baseline.py` 第 206 行注释写作 `# Class imbalance handling (Focal Loss parameters)`，
但其下三行：

```python
'box': 7.5,
'cls': 0.5,
'dfl': 1.5,
```

是 YOLOv8 的**损失权重系数**（框回归 / 分类 / 分布焦点损失），**不是 Focal Loss**。
该注释已在本仓库中更正。若要真正引入 Focal Loss，需改写 ultralytics 的分类损失实现。

---

## 七、后续工作建议（按优先级）

1. **先跑 `tools/verify_cbam.py` 自检**（30 秒）
2. **跑通训练并产出基线结果**（最高优先）
   - 先用 `epochs=30, imgsz=640` 验证链路，再上 `epochs=200`
   - 产出 `best.pt` 与 `results.csv` 后才能谈指标
3. **做 CBAM 消融**（这是任务书 4.3 的核心证据）

   | 实验 | 命令 |
   | --- | --- |
   | 基线 | `python train_baseline.py` |
   | +CBAM(backbone) | `overrides={'cbam':'backbone'}` |
   | +CBAM(neck) | `overrides={'cbam':'neck'}` |
   | +CBAM(all) | `overrides={'cbam':'all'}` |

   四次实验用 `step9_experiment_tracker.py` 自动生成对比表。
4. **补滑窗推理消融**（任务书 4.3 的第二项证据）
   - `sliding_window_inference.py --compare-direct` 同时给出「整图直推」与
     「滑窗融合」的检测数，可直接算召回提升
5. **补横向对比**（任务书 4.4）
   - 至少 YOLOv8n/s/m 三档 + 一个非 YOLO 模型（RT-DETR 在 ultralytics 里可直接调用）
6. **论文中诚实交代数据局限**
   - 训练集经增广扩增至 3840 框，测试集保持原始真实分布（33 框，3 类无样本）
   - 主指标采用 `stable_macro_AP`

---

## 八、仓库说明

本仓库**只收录代码与数据集元数据**，未包含大体量二进制与图片：

| 排除项 | 体积 | 原因 |
| --- | --- | --- |
| `pfmd_tiled_dataset_refined/` 的 images + labels | 3.7 GB | 训练数据，应单独分发 |
| `pfmd_tiled_dataset/` 的 images + labels | 652 MB | 同上 |
| `data/` 原始图片与掩码 | 170 MB | 可由 `data压缩包/` 中的 7z 还原 |
| `yolov8s.pt` | 22 MB | COCO 预训练权重，`YOLO('yolov8s.pt')` 会自动下载 |
| `__pycache__/`、`runs/`、空的训练目录 | — | 缓存与运行产物 |

### 恢复完整数据集

```bash
# 方式一：用原始数据重建（需先解压 data压缩包/ 中的 7z）
#   Defect_images.7z → data/Defect_images/Defect_images/
#   Mask_images.7z   → data/Mask_images/Mask_images/
#   NODefect_images.7z → data/NODefect_images/NODefect_images/
python legacy_steps/step1_integration.py
python legacy_steps/step2_mask_to_yolo.py
python step0_rebuild_tiled_dataset_from_raw.py
python step8_refine_split_and_boost.py
python step10_expand_refined_train_sim.py

# 方式二：直接向作者索取 pfmd_tiled_dataset_refined/ 目录
```

---

## 九、许可

本项目仅用于学习与毕业设计交流，不用于商业用途。
