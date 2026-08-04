# Breast_NAC_pCR_Prediction

- 标签：`N-pCR → 0`，`pCR → 1`
- 临床特征白名单：
  - 因子变量（one-hot）：`Molecular, T, N, HER2`（Molecular 四种：`HR+HER2-` / `HR+HER2+` / `TNBC` / `HER2`）
  - 连续变量（标准化）：`Age, ER, PR, Ki67`
- K 折分层依据由 `--stratify_by` 控制（默认 `Molecular_label`：Molecular×label 联合分层）；划分写入 `kfold_splits.yaml`；超参写入 `config.yaml`
- 入口脚本：[`Classification/main_pcr.py`](Classification/main_pcr.py)
- 使用说明：[`Classification/main_pcr_readme.md`](Classification/main_pcr_readme.md)
- 示例数据：[`example_dataset.csv`](example_dataset.csv) / [`Classification/Example_Dataset_Csv.csv`](Classification/Example_Dataset_Csv.csv)

```bash
cd Classification
python main_pcr.py --mode train --split_mode kfold \
    --csv_path Example_Dataset_Csv.csv --log_root ./logs --exp_name pcr_kfold
```
