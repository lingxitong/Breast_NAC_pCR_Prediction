# Breast_NAC_pCR_Prediction

乳腺新辅助治疗 **pCR 二分类**预测（参考 `BreastRCB-Prognosis` 的 MIL + 临床中期融合框架）。

| 文件 | 说明 |
| --- | --- |
| [`main_pcr.py`](main_pcr.py) | 训练 / 推理入口 |
| [`make_kfold_splits.py`](make_kfold_splits.py) | 独立 K 折划分 |
| [`example_dataset.csv`](example_dataset.csv) | 示例数据 |
| [`Feature_dict.json`](Feature_dict.json) | 字段说明 |

核心逻辑：
- **标签**：`N-pCR → 0`，`pCR → 1`
- **模态**：`pathomic`（病理+临床，默认）/ `pathology`（仅病理）/ `clinical`（仅临床）
- **K 折**：必须先用 `make_kfold_splits.py` 划分；训练强制 `--splits_path` 加载，`main_pcr.py` 不做现场划分
- **临床编码**：因子 `Molecular,T,N,HER2` one-hot；连续 `Age,ER,PR,Ki67` 标准化  
  Molecular 四种：`HR+HER2-` / `HR+HER2+` / `TNBC` / `HER2`

```bash
# 1) 划分
python make_kfold_splits.py --csv_path example_dataset.csv \
    --out_dir ./splits/mol_label_k5 --k 5 --stratify_by Molecular_label

# 2) 基于划分训练（多模态）
python main_pcr.py --mode train --split_mode kfold \
    --csv_path example_dataset.csv \
    --splits_path ./splits/mol_label_k5 \
    --log_root ./logs --exp_name pcr_kfold

# 3) 仅临床信息
python main_pcr.py --mode train --split_mode kfold --clinical_only \
    --csv_path example_dataset.csv \
    --splits_path ./splits/mol_label_k5 \
    --log_root ./logs --exp_name pcr_clinical
```

---

## 一、输入数据格式

见 [`example_dataset.csv`](example_dataset.csv)。

### 必需列

| 列名 | 说明 |
| --- | --- |
| `case_id` | 患者 ID |
| `slide_id` | 切片 ID |
| `slide_feats_path` | 切片特征路径（`.pt` / `.h5`）；**仅临床模式可不依赖特征文件存在** |
| `label` | `N-pCR` / `pCR`，或 `0` / `1` |

### 临床列

| 列名 | 类型 | 编码 | 说明 |
| --- | --- | --- | --- |
| `Molecular` | 因子 | one-hot | `HR+HER2-` / `HR+HER2+` / `TNBC` / `HER2` |
| `T` / `N` / `HER2` | 因子 | one-hot | 分期 / IHC（列名 `HER2` ≠ Molecular 取值 `HER2`） |
| `Age` / `ER` / `PR` / `Ki67` | 连续 | 标准化 | 年龄与免疫组化 |

---

## 二、K 折划分（独立脚本）

```bash
python make_kfold_splits.py \
    --csv_path example_dataset.csv \
    --out_dir ./splits/mol_label_k5 \
    --k 5 --seed 1 --stratify_by Molecular_label
```

输出：

```
splits/mol_label_k5/
  kfold_splits.yaml / .json
  split_config.yaml / config.yaml
  fold_0/split.yaml, train_case_ids.txt, val_case_ids.txt
  fold_1/ ...
```

训练时：

```bash
python main_pcr.py --mode train --split_mode kfold \
    --csv_path example_dataset.csv \
    --splits_path ./splits/mol_label_k5/kfold_splits.yaml \
    ...
```

`--splits_path` 也可直接给划分目录。**kfold 训练时该参数必填**；未提供将直接报错。

### `make_kfold_splits.py --stratify_by`

| 取值 | 含义 |
| --- | --- |
| `Molecular_label`（默认） | Molecular × label 联合分层 |
| `Molecular` | 仅分子分型 |
| `label` | 仅结局 |
| `none` | 不分层 |

失败时按 `Molecular_label → Molecular → label → none` 回退。

---

## 三、模态与模型

| `--modality` | 别名 | 说明 |
| --- | --- | --- |
| `pathomic` | 默认 | WSI MIL + 临床中期融合 |
| `pathology` | `--no-use_clinical` | 仅病理 |
| `clinical` | `--clinical_only` | 仅临床 MLP，不加载 WSI |

**pathomic：**

```
slide bag → MIL → 全局表征 ─┐
临床向量 → MLP → 临床嵌入 ─┴→ fusion → 分类头
```

**clinical：**

```
临床向量 → MLP → logits
```

`--fusion_type`：`concat` / `bilinear` / `gated`（仅 pathomic）

---

## 四、常用命令

```bash
# 基于预划分的多模态 K 折
python main_pcr.py --mode train --split_mode kfold \
    --csv_path example_dataset.csv \
    --splits_path ./splits/mol_label_k5 \
    --log_root ./logs --exp_name pcr_kfold

# 仅临床
python main_pcr.py --mode train --split_mode kfold --clinical_only \
    --csv_path example_dataset.csv \
    --splits_path ./splits/mol_label_k5 \
    --log_root ./logs --exp_name pcr_clinical

# 仅病理
python main_pcr.py --mode train --split_mode kfold --modality pathology \
    --csv_path example_dataset.csv \
    --splits_path ./splits/mol_label_k5 \
    --log_root ./logs --exp_name pcr_path

# 全量训练
python main_pcr.py --mode train --split_mode all_train \
    --csv_path example_dataset.csv --log_root ./logs --exp_name pcr_all

# 推理
python main_pcr.py --mode infer \
    --config ./logs/pcr_kfold/config.yaml \
    --ckpt_path ./logs/pcr_kfold/fold_0/checkpoint_best.pt \
    --csv_path test.csv --save_infer_dir ./infer_pcr
```

---

## 五、主要超参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--modality` | `pathomic` | `pathomic` / `pathology` / `clinical` |
| `--clinical_only` | 关 | 等价 `--modality clinical` |
| `--splits_path` | 无（kfold 必填） | 预划分 yaml/json 或目录 |
| `--split_mode` | `kfold` | `kfold` / `all_train` |
| `--k` | 以划分文件为准 | 训练侧兼容字段 |
| `--model_type` | `abmil` | `abmil` / `mean_mil` / `max_mil` |
| `--fusion_type` | `concat` | 中期融合方式 |
| `--max_epochs` | `50` | 训练轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--gc` | `16` | 梯度累积 |
| `--max_slides_train` | `3` | 训练时最多拼接 slide 数 |

---

## 六、输出与日志

### 训练（kfold）

```
logs/exp_name/
  config.yaml                 # 含 modality / splits_path / stratify_by
  kfold_splits.yaml           # 本实验使用的划分副本
  kfold_summary.yaml
  fold_0/
    split.yaml
    checkpoint_best.pt
    clinical_encoder.json     # pathomic / clinical
    metrics.csv
  ...
```

### 推理

```
infer_dir/
  predictions.csv
  metrics.json
```

---

## 七、依赖

`torch`, `numpy`, `pandas`, `scikit-learn`, `h5py`, `PyYAML`
