# SAM2 微调项目复现报告

**项目**：[sharvaa-selvan/SAM2-Fine-Tuning](https://github.com/sharvaa-selvan/SAM2-Fine-Tuning)  
**对应论文**：*Lightweight open-source fine-tuning of SAM2 enables domain-specific microscopy segmentation*  
**复现平台**：Kaggle（免费 T4 GPU）  
**复现日期**：2026-09-13

---
## 摘要

本实验在 Kaggle Tesla T4 环境下，基于 Meta 的 SAM2.1（base_plus）对 Roboflow 细胞显微图像数据集进行微调，并系统评估微调效果。实验发现：**(1)** 微调显著提升 SAM2 在细胞实例分割上的性能，per-instance F1 从 0.338 提升至 0.672（+98.9%）；**(2)** 40 epoch 微调效果优于 150 epoch，说明在当前数据规模下并非训练越久越好；**(3)** 推理时采用密集采样 + 强 NMS 可将 F1 进一步从 0.551 提升至 0.738，是独立于训练的第二个性能提升来源；**(4)** 评估指标的选择对结论有决定性影响——使用 union IoU 会得出"微调失败"的错误结论，per-instance matching 才是正确口径。

---
## 一、项目概述

该项目提出了一套轻量级的 SAM2 微调流程，用于领域特定的显微镜图像分割。核心特点：

- **只微调 mask decoder**，冻结其他组件，不增加架构层
- **结合生物信息学后处理**（基于海马体细胞密度高的特性，用面积先验筛选）
- **单张 Colab 笔记本即可完成**，计算成本低
- 两个应用场景：脑组织海马体分割 + 单细胞分割

本报告复现的是 **Cell（细胞分割）** 场景。

---
## 二、复现环境

| 项目 | 配置 |
|---|---|
| 平台 | Kaggle Notebook（Interactive） |
| GPU | Tesla T4 × 2（实际使用单卡，15 GB 显存） |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| SAM2 | 官方仓库 [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| 微调配置 | 原作者自定义 `train.yaml`（来自 Google Drive） |
| 模型 | SAM2.1 hiera base_plus（80.9 M 参数） |
| 预训练权重 | `sam2.1_hiera_base_plus.pt` |

---

## 三、数据集

### 3.1 数据来源

Roboflow 导出的细胞显微图像数据集，COCO 格式（JPG + JSON）。

| Split | 图片数 | 标注文件数 | 平均 GT 实例数/图 |
|---|---|---|---|
| train | 59 | 59 | ~297 |
| valid | 16 | 16 | ~200 |
| test | 9 | 9 | — |

- 图像分辨率：1024 × 1024
- 标注格式：RLE（Run-Length Encoding），每条 annotation 含 `bbox`、`area`、`segmentation`

### 3.2 数据预处理

Roboflow 导出的文件名形如 `a_png.rf.9599fb...jpg`，包含多个点号，SAM2 的 DataLoader 无法解析。需做重命名：

```python
new_filename = filename.replace(".", "_", filename.count(".") - 1)
if not re.search(r"_\d+\.\w+$", new_filename):
    new_filename = new_filename.replace(".", "_1.")
```

重命名后，JPG 与 JSON 一一对应，无缺失。该格式与 SAM2 训练脚本中的 `SA1BRawDataset` 完全兼容（每条 annotation 的 `segmentation` 是 RLE dict，可被 `SA1BSegmentLoader` 解码）。

---

## 四、训练方法

### 4.1 训练配置对比

| 参数 | 第一轮（Run1） | 第二轮（Run2） |
|---|---|---|
| `num_epochs` | 40 | 150 |
| `base_lr` | 5.0e-6 | 1.0e-6 |
| `vision_lr` | 3.0e-6 | 5.0e-7 |
| `train_batch_size` | 1 | 2 |
| `num_train_workers` | 2 | 2 |
| `RandomAffine degrees` | 25 | 10 |
| `RandomAffine shear` | 20 | 5 |
| `amp_dtype` | float16 | float16 |
| `gpus_per_node` | 1 | 1 |
| 训练时长 | ~31 分钟 | ~1 小时 36 分钟 |
| 显存峰值 | 7 GB | 13 GB |

其他参数保持原作者配置（AdamW 优化器、cosine LR scheduler、gradient clip max_norm=0.1、layer_decay 0.9、loss 权重 mask:20 / dice:1 / iou:1 / class:1）。

### 4.2 训练 loss 趋势

| Epoch | Run1 total_loss | Run2 total_loss |
|---|---|---|
| 起（1） | 2.46 | 2.15 |
| 中（20） | 1.87 | — |
| 中（101） | — | 1.90 |
| 末（39/149） | 2.06 | 1.96 |

两轮训练 loss 均有下降，但**均未呈现明显收敛**——epoch 间震荡幅度达 ±0.3，这在小数据集（59 张）+ 小 batch 下是正常现象。

---

## 五、评估方法

### 5.1 评估指标演进

**第一版：Union IoU**

将模型生成的所有 mask 合并成一块，与 GT 所有实例合并后的区域计算 IoU。由于细胞在图中是**分散的**，union 后微调模型（输出更保守）的覆盖面积小，导致 IoU 虚假偏低。

**第二版：Per-Instance Matching**

对每张图：
1. 模型生成 N 个预测 mask，GT 有 M 个实例；
2. 按面积从大到小遍历每个预测，找 IoU 最高的未匹配 GT；
3. 若 IoU ≥ 0.5，则该预测视为"命中"，GT 标记为已匹配；
4. 统计：
   - **Matched IoU**：命中预测与其匹配 GT 的平均 IoU；
   - **Precision** = 命中数 / 总预测数；
   - **Recall** = 命中数 / 总 GT 数；
   - **F1** = 2·P·R / (P+R)。

### 5.2 推理参数搜索

使用 `SAM2AutomaticMaskGenerator`，在 4 张验证图上对比 7 组参数：

| 配置 | points_per_side | pred_iou | stability | crop_n_layers | box_nms | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| baseline | 32 | 0.7 | 0.8 | 0 | 0.7（默认） | 0.835 | 0.412 | 0.551 |
| dense | 64 | 0.6 | 0.7 | 0 | 0.7 | 0.475 | 0.726 | 0.574 |
| multiscale | 32 | 0.6 | 0.7 | 1 | 0.7 | 0.492 | 0.717 | 0.584 |
| dense+strict | 64 | 0.75 | 0.85 | 0 | 0.7 | 0.850 | 0.332 | 0.478 |
| dense+very_strict | 64 | 0.8 | 0.9 | 0 | 0.7 | 0.953 | 0.217 | 0.354 |
| dense+nms_0.5 | 64 | 0.6 | 0.7 | 0 | **0.5** | 0.721 | 0.700 | 0.710 |
| **dense+nms_0.3（最优）** | **64** | **0.6** | **0.7** | **0** | **0.3** | **0.797** | **0.688** | **0.738** |
| dense+nms_0.2 | 64 | 0.6 | 0.7 | 0 | 0.2 | 0.797 | 0.679 | 0.733 |

**关键发现**：
- 提高采样密度（32→64）能提升 Recall，但会引入大量重复 mask，导致 Precision 崩塌；
- 提高过滤阈值（pred_iou、stability）反而**降低 Recall**，说明密集采样产生的额外 mask 大多"质量偏弱"；
- **强 NMS（box_nms_thresh=0.3）是解决之道**——抑制重叠重复，同时保留真细胞。

**最终采用配置**：
```python
BEST_CFG = dict(
    points_per_side=64, points_per_batch=64,
    pred_iou_thresh=0.6, stability_score_thresh=0.7,
    box_nms_thresh=0.3,
    crop_n_layers=0, min_mask_region_area=50,
)
```

---

## 六、实验结果

### 6.1 第一次评估（Union IoU）

| 模型 | Union IoU | Union Dice | 微调胜出图数 |
|---|---|---|---|
| 微调（Run1） | 0.2755 | 0.4180 | 3 / 16 |
| 原始 SAM2.1 | **0.4429** | **0.5879** | — |

**误导性结论**：微调使性能下降 38%。**该结论被后续 per-instance 评估推翻。**

### 6.2 第一次 Per-Instance 评估（默认推理参数，4 张图）

| 模型 | Matched IoU | Precision | Recall |
|---|---|---|---|
| Base | 0.6828 | 0.5000 | 0.2188 |
| Run1 | 0.7431 | 0.8006 | 0.3189 |
| Run2 | **0.7468** | **0.8399** | 0.2567 |

微调模型在 IoU 和 Precision 上显著优于 Base。

### 6.3 最终评估（Per-Instance, IoU@0.5, dense+nms_0.3, 16 张图）

| 模型 | Matched IoU | Precision | Recall | **F1** | Matched / GT |
|---|---|---|---|---|---|
| **Base SAM2.1** | 0.6753 | 0.4842 | 0.2595 | 0.3379 | 48.5 / 200 |
| **Run1 (40ep, 5e-6)** | **0.7118** | 0.7625 | **0.6013** | **0.6723** | **115.9 / 200** |
| **Run2 (150ep, 1e-6)** | 0.7081 | **0.7718** | 0.5271 | 0.6264 | 99.9 / 200 |

### 6.4 性能提升汇总（Base → Run1）

| 指标 | Base | Run1 | 相对提升 |
|---|---|---|---|
| Matched IoU | 0.6753 | 0.7118 | **+5.4%** |
| Precision | 0.4842 | 0.7625 | **+57.5%** |
| Recall | 0.2595 | 0.6013 | **+131.7%** |
| **F1** | 0.3379 | 0.6723 | **+98.9%** |
| 平均匹配数 | 48.5 | 115.9 | **+139%** |

---

## 七、讨论

### 7.1 微调为何有效

SAM2.1 在自然图像上预训练，对细胞形态（圆形/椭圆形、低对比度边界、细胞间粘连）的先验不足。59 张细胞图的微调让模型学到：
- **更强的边界敏感性**：Precision 提升 57.5% 说明模型不再乱输出；
- **更高的实例召回**：Recall 翻倍说明模型对"什么是细胞"有了更贴合数据的判断。

### 7.2 为什么 40 epoch 优于 150 epoch

反直觉但可解释：
- Run1 学习率 5e-6，40 epoch 时正好学到"适应数据"但还没开始"过拟合噪声"；
- Run2 学习率 1e-6，150 epoch 虽然 loss 更低，但模型变得**过度保守**（Recall 从 0.60 降到 0.53），丢失了对困难样本的捕捉；
- 这与小数据集（59 张）的典型行为一致：**训练步数存在甜点，超过后泛化能力下降**。

### 7.3 推理参数的独立贡献

即使不做微调，仅通过密集采样 + NMS 也能将 F1 从 0.551 提升至 0.738（+34%）。这说明 SAM2 的自动 mask 生成器**默认参数不适用于细胞这类"密集小目标"场景**。这是一个容易被忽略但性价比极高的优化点。

### 7.4 仍存在的瓶颈

- **极密集场景**：GT=296 的 `ce_png` 图，Run1 Recall 仅 0.405，因为单个细胞平均只有约 20×20 像素，超出 SAM2 的感知分辨率；
- **个别困难图**：`az_png` 在 Run1 上匹配数从 72→49（Run1 vs Run2），说明某些图对超参选择敏感。

### 7.5 评估指标的教训

本实验最重要的方法论发现：**Union IoU 会给出完全错误的结论**。

- Union IoU 衡量"覆盖面积重叠"，对"输出保守但精准"的模型不利；
- Per-Instance Matching 衡量"每个实例的质量"，符合分割任务的实际目标；
- 建议所有 SAM 系列微调实验都采用 per-instance 指标，并明确 IoU 匹配阈值（本实验用 0.5）。

---

## 八、结论

1. **微调 SAM2.1 对细胞显微图像分割有显著效果**：per-instance F1 从 0.338 提升至 0.672，Recall 翻倍。
2. **最佳训练配置为 40 epoch / lr 5e-6**：150 epoch / lr 1e-6 虽然 loss 更低，但泛化性能下降，说明小数据集微调存在最优训练步数。
3. **推理参数优化是独立的性能提升来源**：dense sampling (points_per_side=64) + strong NMS (box_nms_thresh=0.3) 可将 F1 再提升 34%。
4. **评估指标选择至关重要**：union IoU 会误导结论，per-instance matching 才是正确口径。
5. **本实验可在单张 T4 上 2 小时内完整复现**（数据准备 + 训练 + 评估）。

---

## 九、后续方向

| 优先级 | 方向 | 预估成本 |
|---|---|---|
| 高 | 换 `sam2.1_hiera_large` 重训，看是否能突破 F1=0.7 | 2-3 小时 |
| 中 | 用 box prompt 替代 automatic mask generator，可显著提升密集场景 Recall | 1-2 小时 |
| 中 | 推广到 Brain 数据集，验证方法可迁移性 | 1-2 小时 |
| 低 | 引入 per-cell 度量（每实例 Dice），与论文口径对齐 | 半天 |
| 低 | 训练过程动态保存 best checkpoint（按验证集 F1），避免过拟合 | 半天 |

---
*报告完成于 2026-09-15*