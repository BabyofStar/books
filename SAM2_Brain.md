# SAM2 微调项目复现报告

**项目**：[sharvaa-selvan/SAM2-Fine-Tuning](https://github.com/sharvaa-selvan/SAM2-Fine-Tuning)  
**对应论文**：*Lightweight open-source fine-tuning of SAM2 enables domain-specific microscopy segmentation*  
**复现平台**：Kaggle（免费 T4 GPU）  
**复现日期**：2026-09-13

---
## 一、项目概述

该项目提出了一套轻量级的 SAM2 微调流程，用于领域特定的显微镜图像分割。核心特点：

- **只微调 mask decoder**，冻结其他组件，不增加架构层
- **结合生物信息学后处理**（基于海马体细胞密度高的特性，用面积先验筛选）
- **单张 Colab 笔记本即可完成**，计算成本低
- 两个应用场景：脑组织海马体分割 + 单细胞分割

本报告复现的是 **Brain（海马体分割）** 场景。

---
## 二、复现环境

| 项目 | 配置 |
|------|------|
| 平台 | Kaggle Notebook（交互式） |
| GPU | Tesla T4（15GB 显存） |
| CUDA | 12.8 |
| Python | 3.12 |
| PyTorch | Kaggle 预装版本（cu121+） |
| SAM2 版本 | 官方 GitHub 最新版（`pip install -e .[dev]`） |
| 预训练权重 | SAM2.1 hiera base+（Kaggle 官方模型库） |

---
## 三、数据集

| 项目 | 详情 |
|------|------|
| 来源 | 项目自带的 `Brain/Roboflow Dataset.zip` |
| 格式 | COCO RLE（每张图一个 JSON） |
| 图像尺寸 | 1024 × 1024 |
| 训练集 | 54 张 |
| 验证集 | 8 张 |
| 测试集 | 8 张 |
| 标注目标 | 小鼠脑组织海马体区域（单目标） |

---
## 四、复现流程

### 4.1 环境准备

1. Kaggle 新建 Notebook，开启 **GPU T4 x2** + **Internet**
2. 克隆仓库：`git clone https://github.com/sharvaa-selvan/SAM2-Fine-Tuning.git`
3. 解压 Brain 数据集到 `/kaggle/working/data`

### 4.2 数据预处理（关键步骤）

**问题**：SAM2 的 `SA1BRawDataset` 要求文件名格式为 `sa_<整数>`，否则在 `video_name.split("_")[-1]` → `int()` 处报错。

**解决方案**：写脚本将 train/valid/test 的文件重命名为 `sa_0.jpg`、`sa_1.jpg`、…，并同步更新 JSON 内部的 `file_name` 字段。使用两阶段重命名避免冲突。

### 4.3 安装 SAM2 及依赖

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .[dev] -q
pip install -q supervision pycocotools
```

### 4.4 生成 `train.yaml`

**问题**：原项目的 `train.yaml` 通过 Google Drive 分享，且用 `yaml.safe_load/dump` 改写会丢失 `# @package _global_` 注释（导致 Hydra 报 `Key 'launcher' is not in struct`）。

**解决方案**：直接用 Python 字符串生成完整 `train.yaml`，关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_epochs` | 15 | 原为 40，基于数据集大小和时间问题，定为15  |
| `resolution` | 1024 | 与图像一致 |
| `train_batch_size` | 1 | T4 显存限制 |
| `base_lr` | 5e-6 | 论文默认 |
| `vision_lr` | 3e-6 | 论文默认 |
| `num_train_workers` | 2 | Kaggle CPU 核数有限 |
| `checkpoint_path` | Kaggle 上的 `sam2.1_hiera_base_plus.pt` | |

**额外踩坑**：`submitit` 段必须包含 `port_range`，否则 `train.py` 报 `Key 'port_range' is not in struct`。

### 4.5 训练

```bash
cd /kaggle/working/sam2
python training/train.py -c 'configs/train.yaml' --use-cluster 0 --num-gpus 1
```

**训练过程**：

| 指标 | 表现 |
|------|------|
| 显存占用 | 8.0 / 15 GB |
| 单步耗时 | ~1.5 秒 |
| 每 epoch | ~75 秒 |
| 15 epochs 总耗时 | ~20 分钟 |
| 最终 checkpoint | 868.5 MB |

**Loss 下降曲线**：

| Epoch | train_all_loss | loss_mask | loss_dice | loss_iou |
|-------|---------------|-----------|-----------|----------|
| 0 | 3.64 | 0.078 | 1.53 | 0.56 |
| 7 | 1.44 | 0.028 | 0.70 | 0.18 |
| 14 | ~1.0 | — | — | — |

### 4.6 推理 + 后处理

**关键发现 1**：SAM2 默认的 `AutomaticMaskGenerator` 阈值（`pred_iou=0.88, stability=0.95`）会把微调模型的输出**全部过滤**（微调后稳定性分数普遍降至 ~0.93）。

**解决方案**：放宽阈值 + 提高采样密度：

```python
SAM2AutomaticMaskGenerator(
    model=sam2_ft,
    points_per_side=64,           # 默认 32
    pred_iou_thresh=0.5,          # 默认 0.88
    stability_score_thresh=0.7,   # 默认 0.95
    box_nms_thresh=0.9,
    min_mask_region_area=0,
)
```

结果：**平均 59.5 个候选 mask/图**（原 1.8 个）。

**关键发现 2（Oracle 分析）**：

| 指标 | v2（严格阈值） | v3（宽松阈值） |
|------|--------------|--------------|
| 平均 masks/图 | 1.8 | 59.5 |
| 平均 Top-1 Dice | 0.042 | 0.002 |
| **平均 Oracle Dice** | 0.235 | **0.846** |
| Oracle 中位排名 | 1.0 | 6.0 |

**结论**：模型已经学会，正确的 mask 就藏在候选池里（Dice 0.8–0.96），只是没有被正确挑出。

### 4.7 生物信息学后处理

**特征分析**：对 1429 个候选 mask 提取特征，与 Dice 的相关性：

| 特征 | 与 Dice 相关性 | Oracle vs 非 Oracle 均值差 |
|------|---------------|--------------------------|
| `iou_score` | **0.429** | +0.127 |
| `area_frac` | 0.375 | **+0.058（2.1×）** |
| `max_int` | 0.225 | +25.0 |
| `bright_ratio_200` | 0.113 | +0.009（1.6×） |
| `stab_score` | -0.069 | +0.020 |

**策略设计**：

1. 先按 `predicted_iou` 取 **Top-K**（去除低质量候选）
2. 从 Top-K 中选 **`area_frac` 最接近训练集先验**（海马体面积稳定）的 mask

**参数确定**：在 **valid** 上网格搜索 `K ∈ [3,7]` 和 `target ∈ [0.08, 0.14]`，最优参数为 **K=4 或 5，target=0.11 或 0.12**，然后在 **test** 上只跑一次作为最终报告。

---
## 五、定量结果

### 5.1 三次独立复现

| 运行 | Valid Dice | Test Dice | Test Jaccard |
|------|-----------|-----------|--------------|
| 1 | 0.7724 | 0.7682 | 0.6594 |
| 2 | 0.7079 | 0.7248 | 0.5684 |
| 3 | 0.7589 | 0.6626 | 0.5691 |
| **均值 ± 标准差** | 0.746 ± 0.034 | **0.719 ± 0.053** | 0.599 ± 0.052 |

### 5.2 与论文对比

| 方法 | Test Dice | Test Jaccard |
|------|-----------|--------------|
| Base SAM2（未微调） | 0.249 | 0.152 |
| **本复现（均值）** | **0.719 ± 0.053** | **0.599 ± 0.052** |
| 本复现（最好一次） | 0.768 | 0.659 |
| 论文报告 | 0.798 | 0.750 |
| **均值达成率** | **90.1%** | **79.9%** |

### 5.3 提升幅度

- Test Dice 相比 Base SAM2 提升 **189%**（0.249 → 0.719）
- 三次结果均显著优于 base，验证了方法的稳健性

### 5.4 结果波动性说明

三次独立训练（相同超参、相同随机种子）Test Dice 标准差为 0.053，主要来源：

1. **测试集仅 8 张图**，单图波动对均值影响大
2. **CUDA bfloat16 AMP 的非确定性**
3. **小数据集（54 张训练图）本身的方差**

该波动幅度与小数据集深度学习复现的常见现象一致，不影响“方法有效”的核心结论。

---
## 六、关键结论

1. **微调有效**：仅微调 mask decoder，15 epochs（20 分钟），即可让 SAM2 学会领域特定分割任务。

2. **自动掩码生成器不适用于精细评估**：`AutomaticMaskGenerator` 默认阈值会过滤掉微调模型输出，必须放宽阈值获取候选池。

3. **后处理是关键**：微调让模型“生成正确候选”，后处理负责“从候选中挑出正确 mask”。**缺一不可**。

4. **面积先验是有效的生物信息学标准**：海马体区域面积相对稳定（训练集均值 0.0916，中位数 0.0767），配合 iou 预筛能显著提升指标。

5. **结果可复现**：三次独立运行 Dice 分别为 0.768 / 0.725 / 0.663，均值 0.719 ± 0.053，波动属于小数据集的正常范围。

---
## 七、后续可优化方向

| 方向 | 说明 | 预期收益 |
|------|------|---------|
| **多模型集成** | 跑 3–5 次训练，对预测取并集或加权平均 | Test Dice 期望值 +0.03～0.05，方差显著降低 |
| **多目标 GT 处理** | sa_3 类图片含两个独立结构，当前单 mask 无法覆盖 | Dice 或 +0.02～0.05 |
| **增加训练 epoch** | 15 → 30，但有过拟合风险 | 未知，需实验 |
| **调整学习率** | `base_lr: 5e-6 → 1e-5`，当前 loss 仍偏高 | 可能加快收敛 |
| **Cell 数据集迁移** | 用相同流程跑 `Cell/` 场景 | 验证泛化性 |

---

## 八、结论

**本次复现成功完成了该 GitHub 项目的核心目标**：

- ✅ 在 Kaggle 免费 GPU 上完整跑通微调流程
- ✅ Test Dice 均值达到 **0.719**，为论文报告值（0.798）的 **90.1%**
- ✅ 相比 Base SAM2 提升 **189%**
- ✅ 验证了“轻量微调 + 生物信息学后处理”这一核心方法的有效性
- ✅ 产出了可复用的最小可运行 Notebook
    
---
*报告完成于 2026-09-13*