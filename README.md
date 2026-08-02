# Breast_NAC_pCR_Prediction

乳腺新辅助治疗 **pCR 二分类**预测代码（参考 `BreastRCB-Prognosis` 的 MIL + 临床中期融合框架）。

- 标签：`N-pCR → 0`，`pCR → 1`
- 临床特征白名单：`T, N, Age, ER, PR, HER2, Ki67`（**不含 Molecular**）
- 入口脚本：[`Classification/main_pcr.py`](Classification/main_pcr.py)
- 使用说明：[`Classification/main_pcr_readme.md`](Classification/main_pcr_readme.md)
- 示例数据：[`example_dataset.csv`](example_dataset.csv) / [`Classification/Example_Dataset_Csv.csv`](Classification/Example_Dataset_Csv.csv)

```bash
cd Classification
python main_pcr.py --mode train --split_mode kfold \
    --csv_path Example_Dataset_Csv.csv --log_root ./logs --exp_name pcr_kfold
```
