# Breast_NAC_pCR_Prediction

乳腺新辅助治疗 **pCR 二分类**预测（参考 `BreastRCB-Prognosis` 的 MIL + 临床中期融合框架）。

| 文件 | 说明 |
| --- | --- |
| [`main_pcr_train.py`](main_pcr_train.py) | **训练**入口 |
| [`main_pcr_infer.py`](main_pcr_infer.py) | **推理**入口 |
| [`make_kfold_splits.py`](make_kfold_splits.py) | 独立 K 折划分 |
| [`make_kfold_splits_and_test.py`](make_kfold_splits_and_test.py) | 先划独立 test，再对剩余做 K 折 |
| [`mil_models/`](mil_models/) | 自 MIL_BASELINE 扩展的 MIL（`amd_mil` / `wikg_mil` / `gdf_mil`）+ **SDMIL / 三重损失** |
| [`MambaMIL/`](MambaMIL/) | vendored [MambaMIL](https://github.com/isyangshu/MambaMIL)（`mamba_mil` / `trans_mil` / `s4model`） |
| [`requirements.txt`](requirements.txt) | 基础依赖 |
| [`requirements_mamba.txt`](requirements_mamba.txt) / [`scripts/install_mamba.sh`](scripts/install_mamba.sh) | Mamba CUDA 扩展安装 |
| [`example_dataset.csv`](example_dataset.csv) | 示例数据 |
| [`Feature_dict.json`](Feature_dict.json) | 字段说明 |

核心约定：
- **标签**：`N-pCR → 0`，`pCR → 1`
- **模态**：`pathomic`（病理+临床，默认）/ `pathology`（仅病理）/ `clinical`（仅临床）
- **划分**：须先跑 `make_kfold_splits.py` 或 `make_kfold_splits_and_test.py`；训练只认 `--splits_path`，CSV 从划分文件 `meta.csv_path` 读取
- **临床编码**：因子 `Molecular,T,N,HER2` one-hot；连续 `Age,ER,PR,Ki67` 标准化  
  Molecular 四种：`HR+HER2-` / `HR+HER2+` / `TNBC` / `HER2`
- **指标**：AUC / AUPRC / Acc / Balanced Acc / F1 / Precision / Recall（Sensitivity）/ Specificity / PPV / NPV / MCC / TP·TN·FP·FN
- **Bootstrap 95% CI**：终末评估默认 `--n_boot 1000`；单折/OOF/ensemble 对**样本**有放回重采样；K 折平均对**折**有放回重采样（结果写入 `*_ci95_low` / `*_ci95_high`）
- **早停**：有验证集时默认开启（`--patience` / `--early_stop_metric`）

```bash
# 1) 划分（会把 csv 绝对路径写入 meta.csv_path）
#    纯 K 折：
python make_kfold_splits.py --csv_path example_dataset.csv \
    --out_dir ./splits/mol_label_k5 --k 5 --stratify_by Molecular_label
#    或：先划独立 test，再对剩余做 K 折（推荐用于最终外推评估）
python make_kfold_splits_and_test.py --csv_path example_dataset.csv \
    --out_dir ./splits/mol_label_k5_test0.2 \
    --k 5 --test_ratio 0.2 --stratify_by Molecular_label

# 2) 训练（无需再传 --csv_path）
python main_pcr_train.py --split_mode kfold \
    --splits_path ./splits/mol_label_k5_test0.2 \
    --log_root ./logs --exp_name pcr_kfold

# 3) 推理（指定训练日志目录 + 待推理 CSV；可用划出的 test_dataset.csv）
python main_pcr_infer.py \
    --log_dir ./logs/pcr_kfold \
    --csv_path ./splits/mol_label_k5_test0.2/test_dataset.csv \
    --save_dir ./infer/pcr_kfold
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

`.pt` 可为 `float32` tensor，形状 `[N_patch, dim]`；也可为含 `features` 键的 dict。

### 临床列

| 列名 | 类型 | 编码 | 说明 |
| --- | --- | --- | --- |
| `Molecular` | 因子 | one-hot | `HR+HER2-` / `HR+HER2+` / `TNBC` / `HER2` |
| `T` / `N` / `HER2` | 因子 | one-hot | 分期 / IHC（列名 `HER2` ≠ Molecular 取值 `HER2`） |
| `Age` / `ER` / `PR` / `Ki67` | 连续 | 标准化 | 年龄与免疫组化 |

一个 `case_id` 可对应多行（多张 slide）；训练时按患者拼 bag，超过 `--max_slides_train` 则随机采样；验证/推理拼接全部 slide。

---

## 二、K 折划分（独立脚本）

两个划分入口，K 折逻辑一致（总体 + 与总体 fold 对齐的亚组提取）；区别是是否先留出独立 test。

### 2.1 纯 K 折：`make_kfold_splits.py`

对全部样本做 K 折，无独立 test：

```bash
python make_kfold_splits.py \
    --csv_path example_dataset.csv \
    --out_dir ./splits/mol_label_k5 \
    --k 5 --seed 1 --stratify_by Molecular_label
```

输出结构：

```
splits/mol_label_k5/
  kfold_splits.yaml          # meta 含 csv_path / k / stratify_by 等
  fold_0/
    split.yaml
    train_case_ids.txt
    val_case_ids.txt
  fold_1/ ...
  molecular_subgroups/       # 与总体 fold 对齐的亚组划分
    TNBC/
    HER2/
    HR+HER2-/
    HR+HER2+/
```

### 2.2 先划 test 再 K 折：`make_kfold_splits_and_test.py`

先按 `--test_ratio`（患者级）划出独立 test，再对**剩余样本**做与 `make_kfold_splits.py` 相同的 K 折；test / K 折共用同一 `--stratify_by`。独立 test 以 `example_dataset.csv` 同格式落盘，可直接给 `main_pcr_infer.py`。

```bash
python make_kfold_splits_and_test.py \
    --csv_path example_dataset.csv \
    --out_dir ./splits/mol_label_k5_test0.2 \
    --k 5 --seed 1 --test_ratio 0.2 --stratify_by Molecular_label
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--test_ratio` | `0.2` | 独立 test 比例（患者级），须在 `(0, 1)` |
| `--test_csv_name` | `test_dataset.csv` | 输出的 infer 格式 CSV 文件名 |
| `--k` / `--seed` / `--stratify_by` | 同左 | 与 `make_kfold_splits.py` 一致 |

输出结构：

```
splits/mol_label_k5_test0.2/
  split_config.yaml          # test_ratio / n_dev / n_test 等
  test_case_ids.txt          # 独立 test 患者 ID
  test_dataset.csv           # 独立 test，example_dataset.csv 格式（供 infer）
  kfold_splits.yaml          # 仅含剩余样本的 train/val；meta 含 test_* 字段
  fold_0/ ...
  molecular_subgroups/       # 在剩余样本上、与总体 fold 对齐
```

训练仍用 `--splits_path` 指向该目录；推理时把 `--csv_path` 设为 `test_dataset.csv` 即可评估外推性能。

### `--stratify_by`

| 取值 | 含义 |
| --- | --- |
| `Molecular_label`（默认） | Molecular × label 联合分层 |
| `Molecular` | 仅分子分型 |
| `label` | 仅结局 |
| `none` | 不分层 |

---

## 三、训练（`main_pcr_train.py`）

训练**只做训练**，不再内置推理模式。数据入口是 `--splits_path`：

```bash
# 多模态 K 折
python main_pcr_train.py --split_mode kfold \
    --splits_path ./splits/mol_label_k5 \
    --log_root ./logs --exp_name pcr_kfold

# 仅病理
python main_pcr_train.py --split_mode kfold --modality pathology \
    --splits_path ./splits/mol_label_k5 \
    --log_root ./logs --exp_name pcr_path

# 仅临床
python main_pcr_train.py --split_mode kfold --clinical_only \
    --splits_path ./splits/mol_label_k5 \
    --log_root ./logs --exp_name pcr_clinical

# 亚组专训（例如 TNBC）
python main_pcr_train.py --split_mode kfold \
    --splits_path ./splits/mol_label_k5/molecular_subgroups/TNBC \
    --log_root ./logs --exp_name pcr_tnbc

# 全量训练（仍需 splits_path 以定位 CSV；不按折训练）
python main_pcr_train.py --split_mode all_train \
    --splits_path ./splits/mol_label_k5 \
    --log_root ./logs --exp_name pcr_all
```

可选：`--csv_path` 仅在预划分里记录的 CSV 路径失效时，用于覆盖 `meta.csv_path`。

### 早停

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--early_stop` / `--no-early_stop` | 开 | 有验证集时生效；`all_train` 自动跳过 |
| `--patience` | `10` | 连续无提升的 epoch 数 |
| `--min_delta` | `0.0` | 判定提升的最小变化 |
| `--early_stop_metric` | `val_auc` | 还可选 `val_auprc` / `val_loss` / `val_acc` / `val_f1` / `val_recall` 等 |

选模仍优先按验证集 AUC；每折会写 `early_stop.json`。

### 模态与融合

| `--modality` | 别名 | 说明 |
| --- | --- | --- |
| `pathomic` | 默认 | WSI MIL + 临床中期融合 |
| `pathology` | `--no-use_clinical` | 仅病理 |
| `clinical` | `--clinical_only` | 仅临床 MLP，不加载 WSI |

```
# pathomic
slide bag → MIL → 全局表征 ─┐
临床向量 → MLP → 临床嵌入 ─┴→ fusion → 分类头

# clinical
临床向量 → MLP → logits
```

`--fusion_type`：`concat` / `bilinear` / `gated`（仅 pathomic）

`mamba_mil` / `trans_mil` / `s4model` 同样可作为 MIL backbone 接临床中期融合（内部表征维固定 512）。源码已 vendored 于 [`MambaMIL/`](MambaMIL/)；首次使用前需编译安装 CUDA 扩展：

```bash
# 建议环境与上游一致：CUDA 11.8 / Python 3.10 / PyTorch 2.0.x
pip install -r requirements.txt
bash scripts/install_mamba.sh
# 或：pip install -r requirements_mamba.txt && pip install ./MambaMIL/mamba
```

自 [`MIL_BASELINE`](../MIL_BASELINE) 扩展的 `amd_mil` / `wikg_mil` / `gdf_mil` 同样截到 bag 表征后接临床中期融合；源码在 [`mil_models/`](mil_models/)。依赖：

```bash
pip install einops torch_geometric   # amd / wikg+gdf
```

| `--model_type` | 来源 | bag 维 | 额外依赖 |
| --- | --- | --- | --- |
| `abmil` / `mean_mil` / `max_mil` | 本地 | `hidden_dim` | — |
| `mamba_mil` / `trans_mil` / `s4model` | `MambaMIL/` | 512 | mamba CUDA 扩展（mamba） |
| `amd_mil` | `mil_models/AMD_MIL` | `amd_embed_dim`（默认 `hidden_dim`） | `einops` |
| `wikg_mil` | `mil_models/WIKG_MIL` | `wikg_dim_hidden`（默认 `hidden_dim`） | `torch_geometric` |
| `gdf_mil` | `mil_models/GDF_MIL` | `gdf_out_dim`（默认 128） | `torch_geometric` |
| `sdmil` | `mil_models/SDMIL` | `hidden_dim` | —（默认开启三重损失） |

常用超参：`--amd_agent_num`、`--wikg_topk` / `--wikg_agg_type`、`--gdf_k_components` / `--gdf_k_neighbors`。

仅临床模式 `modality=clinical` 仍走独立 MLP，与 `model_type` 无关；**不支持** `sdmil` / 三重损失。

### SDMIL + 三重损失（创新模块）

核心思想：解耦「跨分型通用预后特征」与「分型特异预后特征」。

```
bag → MIL backbone → h ─┬→ ProjectionHead → z ──→ L_Global / L_Intra（Memory Bank）
                        │                   └→ 对齐可学习原型 C_k → L_Inter
                        └→ (+ clinical fusion) → CE
```

总损失：`L = L_CE + α L_Global + β L_Intra + γ L_Inter`

| 损失 | 作用 |
| --- | --- |
| `L_Global` | 忽略分型，按 pCR 标签做 SupCon（跨分型不变性） |
| `L_Intra` | 仅同分子分型内对比（分型内特异性 / 难例负样本） |
| `L_Inter` | 对齐自身分型原型、排斥其他原型（保留生物学分型差异） |

分阶段训练：
1. **Warm-up**（前 `--triple_warmup_epochs`，默认 20）：`CE + α L_Global`
2. **Fine-tune**：引入 `L_Intra/L_Inter`，学习率 × `--triple_finetune_lr_scale`（默认 0.1）

```bash
# 推荐：SDMIL（默认 abmil 聚合 + 三重损失）
python main_pcr_train.py --split_mode kfold \
    --splits_path ./splits/mol_label_k5_test0.2 \
    --model_type sdmil --modality pathomic \
    --triple_alpha 0.5 --triple_beta 0.5 --triple_gamma 0.5 \
    --triple_warmup_epochs 20 --triple_bank_size 256 \
    --log_root ./logs --exp_name sdmil_triple

# 也可把三重损失挂到其他 backbone（会自动换上投影头+原型）
python main_pcr_train.py --splits_path ./splits/mol_label_k5_test0.2 \
    --model_type amd_mil --use_triple_loss \
    --log_root ./logs --exp_name amd_triple
```

相关超参：`--sdmil_base`、`--triple_temperature`、`--triple_proj_dim`、`--triple_bank_size`、`--no-use_triple_loss`。

### 主要超参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--splits_path` | **必填** | 预划分 yaml/json 或目录 |
| `--split_mode` | `kfold` | `kfold` / `all_train` |
| `--modality` | `pathomic` | 见上表 |
| `--model_type` | `abmil` | 见上表（含 amd/wikg/gdf） |
| `--fusion_type` | `concat` | 中期融合 |
| `--max_epochs` | `50` | 最大轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--gc` | `16` | 梯度累积 |
| `--max_slides_train` | `3` | 训练时最多拼接 slide 数 |
| `--in_dim` | 自动推断 | 特征维度；`<=0` 时从 `.pt` 读取 |
| `--n_boot` | `1000` | bootstrap 次数（`<=0` 关闭 CI） |
| `--bootstrap_ci` | `0.95` | 置信水平 |

### 训练日志结构

```
logs/exp_name/
  config.yaml / config.json
  kfold_splits.yaml           # 本实验使用的划分副本
  kfold_summary.yaml          # 含 mean_val_metrics（折级 bootstrap CI）
  mean_val_metrics.yaml       # K 折指标均值 ± 95% CI
  oof_predictions.csv         # 各折验证集拼接
  oof_metrics.json            # OOF 点估计 + 样本级 95% CI
  fold_0/
    split.yaml
    checkpoint_best.pt
    checkpoint_last.pt
    clinical_encoder.json     # pathomic / clinical 时有
    metrics.csv               # 含完整 train_/val_ 指标列
    early_stop.json
    val_predictions.csv
    val_metrics.json          # 含样本级 bootstrap 95% CI
  fold_1/ ...
```

---

## 四、推理（`main_pcr_infer.py`）

输入训练日志目录 + 待推理 CSV + 结果保存目录，自动加载各折权重完成推理：

```bash
python main_pcr_infer.py \
    --log_dir ./logs/pcr_kfold \
    --csv_path example_dataset.csv \
    --save_dir ./infer/pcr_kfold
```

| 参数 | 说明 |
| --- | --- |
| `--log_dir` | 训练日志目录（需含 `config.yaml` 与 `fold_*/checkpoint_*.pt`） |
| `--csv_path` | 与 `example_dataset.csv` 同格式 |
| `--save_dir` | 推理结果目录 |
| `--ckpt_name` | 优先权重名，默认 `checkpoint_best.pt` |
| `--config` | 可选，覆盖日志内部分超参 |

行为：
1. 读取 `log_dir/config.yaml`，发现全部 fold 权重  
2. 对 CSV 全量推理，逐折落盘预测与指标  
3. 多折时对 `prob_pCR` 取均值 → ensemble  
4. 若日志内有 `kfold_splits.yaml` 且 case 可对齐 → 额外输出 OOF

### 推理输出结构

```
infer_dir/
  fold_0/
    predictions.csv
    infer_predictions.csv
    infer_metrics.json / .yaml
  fold_1/ ...
  predictions_ensemble.csv
  ensemble_metrics.json / .yaml
  predictions_oof.csv              # 可选
  oof_metrics.json / .yaml         # 可选
  infer_summary.yaml / .json
```

---

## 五、端到端示例

假设工程根目录为仓库上级，特征已按绝对路径写在 CSV 中：

```bash
# 划分（含独立 test）
python make_kfold_splits_and_test.py \
    --csv_path example_dataset.csv \
    --out_dir ../data_splits \
    --k 5 --seed 1 --test_ratio 0.2 --stratify_by Molecular_label

# 仅病理、ABMIL、短训冒烟
python main_pcr_train.py \
    --splits_path ../data_splits/kfold_splits.yaml \
    --log_root ../logs --exp_name abmil_pathology_e2 \
    --split_mode kfold \
    --model_type abmil \
    --modality pathology \
    --max_epochs 2

# 用独立 test CSV 做推理
python main_pcr_infer.py \
    --log_dir ../logs/abmil_pathology_e2 \
    --csv_path ../data_splits/test_dataset.csv \
    --save_dir ../infer_ans
```

---

## 六、依赖

基础：`torch`, `numpy`, `pandas`, `scikit-learn`, `h5py`, `PyYAML`

扩展模型（按需）：`einops`（`amd_mil`）、`torch_geometric`（`wikg_mil` / `gdf_mil`）
