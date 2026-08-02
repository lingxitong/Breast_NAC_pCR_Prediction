# main_pcr.py 使用说明

基于 `BreastRCB-Prognosis` 中期融合框架改造的单脚本工具，用于**乳腺新辅助治疗 pCR 二分类**，支持训练（K 折 / 全量）与推理，并支持 **WSI 特征 + 临床信息的中期融合**。

核心建模逻辑：
- **二分类**：`N-pCR → 0`，`pCR → 1`（CSV 中也可直接写 `0/1`）。
- **损失**：CrossEntropy。
- **评价指标**：AUC / Accuracy / F1 / Sensitivity / Specificity；K 折按 **val AUC** 选模。
- **多 slide 拼 bag**：一个患者（`case_id`）可能有多张 slide。训练时拼接；若 `> max_slides_train` 则随机抽取；推理时拼接全部。
- **临床中期融合**（默认开启）：仅使用白名单列 `T, N, Age, ER, PR, HER2, Ki67`，**不使用 Molecular**。

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

| 列名 | 说明 |
| --- | --- |
| `T` | 临床 T 分期 |
| `N` | 临床 N 分期 |
| `Age` | 年龄 |
| `ER` / `PR` / `HER2` / `Ki67` | 免疫组化相关 |

> **注意**：即使 CSV 中含有 `Molecular` / `Molecular_subtype`，脚本也会忽略，不进入临床融合。

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
    --config ./logs/pcr_kfold/config.json \
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
  config.json
  kfold_summary.json
  fold_0/
    checkpoint_best.pt      # 按 val AUC
    checkpoint_last.pt
    clinical_encoder.json
    metrics.csv
    history.json
  fold_1/ ...
```

### 推理

```
infer_dir/
  predictions.csv   # case_id, label, prob_pCR, pred
  metrics.json      # auc/acc/f1/...
```

---

## 六、依赖

`torch`, `numpy`, `pandas`, `scikit-learn`, `h5py`
