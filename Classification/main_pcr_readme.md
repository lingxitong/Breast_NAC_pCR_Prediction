# main_pcr.py 使用说明

基于 `BreastRCB-Prognosis` 中期融合框架改造的单脚本工具，用于**乳腺新辅助治疗 pCR 二分类**，支持训练（K 折 / 全量）与推理，并支持 **WSI 特征 + 临床信息的中期融合**。

核心建模逻辑：
- **二分类**：`N-pCR → 0`，`pCR → 1`（CSV 中也可直接写 `0/1`）。
- **损失**：CrossEntropy。
- **评价指标**：AUC / Accuracy / F1 / Sensitivity / Specificity；K 折按 **val AUC** 选模。
- **多 slide 拼 bag**：一个患者（`case_id`）可能有多张 slide。训练时拼接；若 `> max_slides_train` 则随机抽取；推理时拼接全部。
- **临床中期融合**（默认开启）：白名单列 `Molecular, T, N, Age, ER, PR, HER2, Ki67`。
  - **因子变量**（one-hot）：`Molecular, T, N, HER2`
  - **连续变量**（z-score 标准化）：`Age, ER, PR, Ki67`
- **K 折划分**：由 `--stratify_by` 指定分层依据（默认 `Molecular_label`：按 Molecular 与 label 联合分层）。划分结果写入 `kfold_splits.yaml` 与各 `fold_*/split.yaml`。
- **超参数日志**：训练时保存为 `config.yaml`（含 `stratify_by`）；推理 `--config` 支持 yaml/json。

---

## 一、输入数据格式

输入 CSV 格式见 `Example_Dataset_Csv.csv`。

### 必需列

| 列名 | 说明 |
| --- | --- |
| `case_id` | 患者 ID（同一患者多张 slide 共用） |
| `slide_id` | 切片 ID |
| `slide_feats_path` | 该切片特征文件路径（`.pt` 或 `.h5`） |
| `label` | `N-pCR` / `pCR`，或已映射的 `0` / `1` |

### 临床列（白名单，可选但推荐）

| 列名 | 类型 | 编码 | 说明 |
| --- | --- | --- | --- |
| `Molecular` | 因子 | one-hot | 分子分型（四种）：`HR+HER2-` / `HR+HER2+` / `TNBC` / `HER2` |
| `T` | 因子 | one-hot | 临床 T 分期 |
| `N` | 因子 | one-hot | 临床 N 分期 |
| `HER2` | 因子 | one-hot | HER2 状态（IHC 等级等） |
| `Age` | 连续 | 标准化 | 年龄 |
| `ER` | 连续 | 标准化 | ER 表达 |
| `PR` | 连续 | 标准化 | PR 表达 |
| `Ki67` | 连续 | 标准化 | Ki67 指数 |

> **编码说明**：因子变量按训练折类别表做 one-hot（含 `missing`）；连续变量用训练集均值/标准差做 z-score，缺失填均值。`Molecular` 预置四类（`HR+HER2-` / `HR+HER2+` / `TNBC` / `HER2`），即使某折未出现也会保留对应维度。注意列名 `HER2` 是 IHC 因子变量，与 Molecular 取值 `HER2`（HER2 富集型）含义不同。

特征文件：
- `.pt`：`torch.Tensor` / `ndarray`，形状 `[N_patch, dim]`；或含 `features` 键的 dict。
- `.h5`：默认键名 `features`，形状 `[N_patch, dim]`（可用 `--feat_key` 修改）。

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

融合方式（`--fusion_type`）：`concat`（默认）/ `bilinear` / `gated`。  
关闭临床融合：`--no-use_clinical`。

---

## 三、常用命令

```bash
# K 折交叉验证
python main_pcr.py --mode train --split_mode kfold \
    --csv_path Example_Dataset_Csv.csv --log_root ./logs --exp_name pcr_kfold

# 全量训练
python main_pcr.py --mode train --split_mode all_train \
    --csv_path Example_Dataset_Csv.csv --log_root ./logs --exp_name pcr_all

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
| `--k` | `5` | 折数（按 case 分层） |
| `--stratify_by` | `Molecular_label` | K 折分层依据：`Molecular_label` / `Molecular` / `label` / `none` |
| `--model_type` | `abmil` | `abmil` / `mean_mil` / `max_mil` |
| `--fusion_type` | `concat` | 中期融合方式 |
| `--use_clinical` | 开启 | 加 `--no-use_clinical` 关闭 |
| `--max_epochs` | `50` | 训练轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--gc` | `16` | 梯度累积步数 |
| `--max_slides_train` | `3` | 训练时最多拼接 slide 数 |
| `--in_dim` | `-1` | <=0 时从特征文件自动推断 |

---

## 五、输出

### 训练（kfold）

```
logs/exp_name/
  config.yaml               # 超参数（主格式）
  config.json               # 同步备份
  kfold_splits.yaml         # 全部分折 case_id / Molecular / label 分布
  kfold_splits.json
  kfold_summary.yaml
  kfold_summary.json
  fold_0/
    split.yaml              # 该折 train/val 划分明细
    train_case_ids.txt
    val_case_ids.txt
    checkpoint_best.pt      # 按 val AUC
    checkpoint_last.pt
    clinical_encoder.json
    metrics.csv
    history.json
  fold_1/ ...
```

### `--stratify_by` 分层依据

| 取值 | 含义 |
| --- | --- |
| `Molecular_label`（默认） | 按 `Molecular` 与 `label` 联合分层（组合键 `Molecular\|y{0/1}`） |
| `Molecular` | 仅按分子分型分层 |
| `label` | 仅按 pCR / N-pCR 分层 |
| `none` | 不分层，普通随机 KFold |

若请求策略因某联合类样本过少失败，会按  
`Molecular_label → Molecular → label → none` 回退。  
日志中：`stratify_by_requested` 为超参请求值，`stratify_by` 为实际使用值。

### 推理

```
infer_dir/
  predictions.csv   # case_id, label, prob_pCR, pred
  metrics.json      # auc/acc/f1/...
```

---

## 六、依赖

`torch`, `numpy`, `pandas`, `scikit-learn`, `h5py`, `PyYAML`
