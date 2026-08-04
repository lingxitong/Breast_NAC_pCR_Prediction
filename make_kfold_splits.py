#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_kfold_splits.py
====================
独立的 K 折划分脚本。只负责按指定分层依据生成划分并落盘，
训练时可通过 main_pcr.py --splits_path 直接加载，无需重新划分。

输出目录结构：
  {out_dir}/
    kfold_splits.yaml / .json
    split_config.yaml           # 本次划分使用的参数
    fold_0/
      split.yaml
      train_case_ids.txt
      val_case_ids.txt
    fold_1/ ...

示例：
  python make_kfold_splits.py \\
      --csv_path example_dataset.csv \\
      --out_dir ./splits/mol_label_k5 \\
      --k 5 --seed 1 --stratify_by Molecular_label

  # 训练时复用：
  python main_pcr.py --mode train --split_mode kfold \\
      --csv_path example_dataset.csv \\
      --splits_path ./splits/mol_label_k5/kfold_splits.yaml \\
      --log_root ./logs --exp_name pcr_from_splits
"""

from __future__ import print_function

import argparse
import os

from main_pcr import (
    STRATIFY_BY_CHOICES,
    build_fold_split_record,
    build_patient_table,
    make_kfold_splits,
    normalize_stratify_by,
    read_csv_smart,
    save_kfold_splits,
    save_run_config,
    set_seed,
)


def get_args():
    p = argparse.ArgumentParser(description="独立生成 K 折划分（按 Molecular/label 等分层）")
    p.add_argument("--csv_path", type=str, required=True, help="输入 CSV（见 example_dataset.csv）")
    p.add_argument("--out_dir", type=str, required=True, help="划分结果输出目录")
    p.add_argument("--k", type=int, default=5, help="折数（默认 5）")
    p.add_argument("--seed", type=int, default=1, help="随机种子")
    p.add_argument(
        "--stratify_by",
        type=str,
        default="Molecular_label",
        choices=list(STRATIFY_BY_CHOICES),
        help="分层依据：Molecular_label / Molecular / label / none",
    )
    p.add_argument("--label_col", type=str, default="label", help="标签列名")
    p.add_argument(
        "--feat_path_col", type=str, default=None,
        help="特征路径列（可选；划分本身不依赖特征文件是否存在）",
    )
    p.add_argument(
        "--require_feats", action=argparse.BooleanOptionalAction, default=False,
        help="是否要求 CSV 中存在可用的特征路径列（默认不要求）",
    )
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    cfg = {
        "k": int(args.k),
        "seed": int(args.seed),
        "stratify_by": normalize_stratify_by(args.stratify_by),
        "label_col": args.label_col,
        "feat_path_col": args.feat_path_col,
        "csv_path": os.path.abspath(args.csv_path),
    }

    df = read_csv_smart(args.csv_path)
    pt, clinical_cols, feat_col = build_patient_table(
        df,
        label_col=args.label_col,
        feat_path_col=args.feat_path_col,
        require_feats=bool(args.require_feats),
    )
    cfg["feat_path_col"] = feat_col
    cfg["n_cases"] = int(len(pt))
    cfg["clinical_cols"] = list(clinical_cols)

    print(
        f"患者数: {len(pt)}, pCR(1)={int((pt['y'] == 1).sum())}, "
        f"N-pCR(0)={int((pt['y'] == 0).sum())}, "
        f"stratify_by={cfg['stratify_by']}, k={cfg['k']}, seed={cfg['seed']}"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    save_run_config(cfg, args.out_dir)
    # 额外以更直观的名字保留一份
    split_cfg_path = os.path.join(args.out_dir, "split_config.yaml")
    if os.path.isfile(os.path.join(args.out_dir, "config.yaml")):
        # save_run_config 已写 config.yaml；再写 split_config.yaml 便于辨认
        import shutil
        shutil.copy2(os.path.join(args.out_dir, "config.yaml"), split_cfg_path)

    splits, split_meta = make_kfold_splits(pt, cfg)
    fold_records = [
        build_fold_split_record(pt, fold, tr_idx, va_idx, split_meta["stratify_by"])
        for fold, (tr_idx, va_idx) in enumerate(splits)
    ]
    split_meta["csv_path"] = cfg["csv_path"]
    split_meta["label_col"] = args.label_col
    splits_path = save_kfold_splits(args.out_dir, split_meta, fold_records)

    print("\n===== K 折划分完成 =====")
    print(f"stratify_by_requested: {split_meta.get('stratify_by_requested')}")
    print(f"stratify_by (actual):  {split_meta.get('stratify_by')}")
    print(f"输出: {os.path.abspath(args.out_dir)}")
    print(f"训练时可指定: --splits_path {os.path.abspath(splits_path)}")
    for rec in fold_records:
        print(
            f"  fold {rec['fold']}: n_train={rec['n_train']}, n_val={rec['n_val']}, "
            f"val_mol={rec['val_molecular']}, val_label={rec['val_label']}"
        )


if __name__ == "__main__":
    main()
