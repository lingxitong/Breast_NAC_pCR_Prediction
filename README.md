# Breast_NAC_pCR_Prediction

乳腺新辅助治疗 **pCR 二分类**预测（参考 `BreastRCB-Prognosis` 的 MIL + 临床中期融合框架）。

入口脚本：[`main_pcr.py`](main_pcr.py)  
示例数据：[`example_dataset.csv`](example_dataset.csv)  
字段说明：[`Feature_dict.json`](Feature_dict.json)

核心逻辑：
- **标签**：`N-pCR → 0`，`pCR → 1`（CSV 也可直接写 `0/1`）
- **损失**：CrossEntropy
- **指标**：AUC / Accuracy / F1 / Sensitivity / Specificity；K 折按 **val AUC** 选模
- **多 slide 拼 bag**：同一 `case_id` 可有多张 slide；训练时拼接，超过 `max_slides_train` 则随机抽取；推理拼接全部
- **临床中期融合**（默认开启）
  - 因子变量（one-hot）：`Molecular, T, N, HER2`
  - 连续变量（标准化）：`Age, ER, PR, Ki67`
  - Molecular 四种：`HR+HER2-` / `HR+HER2+` / `TNBC` / `HER2`
- **K 折分层**：`--stratify_by`（默认 `Molecular_label`）；划分写入 `kfold_splits.yaml`；超参写入 `config.yaml`

```bash
python main_pcr.py --mode train --split_mode kfold \
    --csv_path example_dataset.csv --log_root ./logs --exp_name pcr_kfold
```

---

## 一、输入数据格式

见 [`example_dataset.csv`](example_dataset.csv)。

### 必需列

| 列名 | 说明 |
| --- | --- |
| `case_id` | 患者 ID（同一患者多张 slide 共用） |
| `slide_id` | 切片 ID |
| `slide_feats_path` | 该切片特征文件路径（`.pt` 或 `.h5`） |
| `label` | `N-pCR` / `pCR`，或 `0` / `1` |

### 临床列（白名单，可选但推荐）

| 列名 | 类型 | 编码 | 说明 |
| --- | --- | --- | --- |
| `Molecular` | 因子 | one-hot | 分子分型：`HR+HER2-` / `HR+HER2+` / `TNBC` / `HER2` |
| `T` | 因子 | one-hot | 临床 T 分期 |
| `N` | 因子 | one-hot | 临床 N 分期 |
| `HER2` | 因子 | one-hot | HER2 IHC 等级（与 Molecular 取值 `HER2` 含义不同） |
| `Age` | 连续 | 标准化 | 年龄 |
| `ER` | 连续 | 标准化 | ER 表达 |
| `PR` | 连续 | 标准化 | PR 表达 |
| `Ki67` | 连续 | 标准化 | Ki67 指数 |

因子变量按训练折类别表 one-hot（含 `missing`）；连续变量用训练集均值/标准差做 z-score。`Molecular` 预置四类，即使某折未出现也保留维度。

特征文件：
- `.pt`：`Tensor` / `ndarray`，形状 `[N_patch, dim]`；或含 `features` 键的 dict
- `.h5`：默认键名 `features`（可用 `--feat_key` 修改）

---

## 二、模型结构（中期融合）

```
slide bag [N, in_dim]
    ↓
MIL 聚合器 (abmil / mean_mil / max_mil)
    ↓
全局表征 [1, hidden_dim]
    ↓                          临床向量 [clinical_in_dim]
    ↓                                ↓
    └──────── fusion_type ──── 临床 MLP → 临床嵌入
                    ↓
              融合表征 [1, hidden_dim]
                    ↓
              分类头 → logits [2]
```

`--fusion_type`：`concat`（默认）/ `bilinear` / `gated`  
关闭临床融合：`--no-use_clinical`

---

## 三、常用命令

```bash
# K 折交叉验证
python main_pcr.py --mode train --split_mode kfold \
    --csv_path example_dataset.csv --log_root ./logs --exp_name pcr_kfold

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

## 四、主要超参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--mode` | `train` | `train` / `infer` |
| `--split_mode` | `kfold` | `kfold` / `all_train` |
| `--k` | `5` | 折数（按 case） |
| `--stratify_by` | `Molecular_label` | `Molecular_label` / `Molecular` / `label` / `none` |
| `--model_type` | `abmil` | `abmil` / `mean_mil` / `max_mil` |
| `--fusion_type` | `concat` | 中期融合方式 |
| `--use_clinical` | 开启 | `--no-use_clinical` 关闭 |
| `--max_epochs` | `50` | 训练轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--gc` | `16` | 梯度累积步数 |
| `--max_slides_train` | `3` | 训练时最多拼接 slide 数 |
| `--in_dim` | `-1` | `<=0` 时从特征文件自动推断 |

### `--stratify_by`

| 取值 | 含义 |
| --- | --- |
| `Molecular_label`（默认） | Molecular 与 label 联合分层（`Molecular\|y{0/1}`） |
| `Molecular` | 仅分子分型 |
| `label` | 仅 pCR / N-pCR |
| `none` | 不分层，随机 KFold |

失败时按 `Molecular_label → Molecular → label → none` 回退。日志中 `stratify_by_requested` 为请求值，`stratify_by` 为实际值。

---

## 五、输出与日志

### 训练（kfold）

```
logs/exp_name/
  config.yaml / config.json
  kfold_splits.yaml / .json
  kfold_summary.yaml / .json
  fold_0/
    split.yaml
    train_case_ids.txt / val_case_ids.txt
    checkpoint_best.pt      # 按 val AUC
    checkpoint_last.pt
    clinical_encoder.json
    metrics.csv / history.json
  fold_1/ ...
```

### 推理

```
infer_dir/
  predictions.csv   # case_id, label, prob_pCR, pred
  metrics.json
```

---

## 六、依赖

`torch`, `numpy`, `pandas`, `scikit-learn`, `h5py`, `PyYAML`
