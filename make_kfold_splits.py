#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_kfold_splits.py
====================
基于总体划分，提取各亚组的划分结果。
确保总体和亚组的 fold 划分完全一致，可公平比较。

输出目录结构：
  {out_dir}/
    kfold_splits.yaml              # 总体划分
    split_config.yaml
    fold_0/...
    molecular_subgroups/
      TNBC/
        kfold_splits.yaml          # 从总体 fold 0 提取的 TNBC 样本
        fold_0/...                 # 与总体 fold 0 的划分一致
      HER2/...
      HR+HER2-/...
      HR+HER2+/...

示例：
  python make_kfold_splits.py \\
      --csv_path example_dataset.csv \\
      --out_dir ./splits/k5_molecular_label \\
      --k 5 --seed 1 --stratify_by Molecular_label
"""

from __future__ import print_function

import argparse
import os
import json
import yaml
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed=1):
    import random
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_csv_smart(path):
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
    for enc in encodings:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法读取文件: {path}")


LABEL_MAP = {
    "n-pcr": 0, "n_pcr": 0, "npcr": 0, "0": 0, 0: 0,
    "pcr": 1, "1": 1, 1: 1,
}


def map_label(v):
    if pd.isna(v):
        raise ValueError("label 存在缺失值")
    if isinstance(v, (int, np.integer)):
        key = int(v)
    else:
        key = str(v).strip()
        key_lower = key.lower().replace(" ", "")
        if key_lower in LABEL_MAP:
            return LABEL_MAP[key_lower]
        if key_lower.replace("-", "") in LABEL_MAP:
            return LABEL_MAP[key_lower.replace("-", "")]
        try:
            key = int(float(key))
        except Exception:
            raise ValueError(f"无法解析 label 值: {v!r}")
    if key not in (0, 1):
        raise ValueError(f"label 必须为 0/1 或 N-pCR/pCR，得到: {v!r}")
    return int(key)


def build_patient_table(df, label_col="label"):
    """构建患者级表"""
    if label_col not in df.columns:
        raise KeyError(f"CSV 缺少标签列 {label_col}")

    df = df.copy()
    df["y"] = df[label_col].map(map_label)
    df = df.dropna(subset=["case_id", "y"])

    # 获取临床列
    clinical_cols = [c for c in ["T", "N", "Age", "ER", "PR", "HER2", "Ki67", "Molecular"]
                     if c in df.columns and c not in ["case_id", "slide_id", "label", "y"]]

    records = []
    for case_id, g in df.groupby("case_id", sort=False):
        labels = g["y"].astype(int).tolist()
        if len(set(labels)) != 1:
            raise ValueError(f"同一 case_id={case_id} 存在不一致 label: {labels}")

        rec = {
            "case_id": str(case_id),
            "y": int(labels[0]),
        }
        for col in clinical_cols:
            rec[col] = g[col].iloc[0] if col in g.columns else None
        records.append(rec)

    pt = pd.DataFrame(records).reset_index(drop=True)
    return pt, clinical_cols


def get_subgroup_molecular_map():
    """定义亚型到 Molecular 列值的映射"""
    return {
        "TNBC": ["TNBC", "tnbc", "Triple Negative", "triple negative"],
        "HER2": ["HER2", "HER2+", "HER2 enriched", "HER2-positive"],
        "HR+HER2-": [
            "HR+HER2-", "Luminal", "Luminal A", "Luminal B",
            "Luminal A/B", "HR+", "ER+PR+HER2-"
        ],
        "HR+HER2+": [
            "HR+HER2+", "Luminal HER2+", "ER+PR+HER2+",
            "Luminal B HER2+", "HR+HER2 positive"
        ],
    }


def save_fold_splits(out_dir, meta, fold_records):
    """保存 K 折划分"""
    os.makedirs(out_dir, exist_ok=True)

    splits_data = {"meta": meta, "folds": fold_records}
    splits_path = os.path.join(out_dir, "kfold_splits.yaml")
    with open(splits_path, "w", encoding="utf-8") as f:
        yaml.dump(splits_data, f, default_flow_style=False, allow_unicode=True)

    for rec in fold_records:
        fold_dir = os.path.join(out_dir, f"fold_{rec['fold']}")
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir, "split.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(rec, f, default_flow_style=False, allow_unicode=True)

        with open(os.path.join(fold_dir, "train_case_ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rec["train_case_ids"]))

        with open(os.path.join(fold_dir, "val_case_ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rec["val_case_ids"]))

    print(f"✅ 划分已保存: {splits_path}")
    return splits_path


def extract_subgroup_folds(pt_df, overall_folds, subgroup_name):
    """从总体划分中提取亚组划分"""
    mapping = get_subgroup_molecular_map()
    molecular_values = mapping.get(subgroup_name, [])

    # 获取亚组样本的索引
    subgroup_indices = []
    for idx, row in pt_df.iterrows():
        mol = str(row.get("Molecular", "")).strip()
        if mol in molecular_values:
            subgroup_indices.append(idx)

    subgroup_indices_set = set(subgroup_indices)

    # 从总体划分中提取亚组划分
    subgroup_folds = []
    for train_idx, val_idx in overall_folds:
        # 只保留亚组样本
        sub_train_idx = [i for i in train_idx if i in subgroup_indices_set]
        sub_val_idx = [i for i in val_idx if i in subgroup_indices_set]
        subgroup_folds.append((sub_train_idx, sub_val_idx))

    return subgroup_folds, len(subgroup_indices)


def build_fold_record(pt_df, fold_idx, train_idx, val_idx, split_type="overall", subgroup_name=None):
    """构建 fold 记录"""
    train_case_ids = pt_df.iloc[train_idx]["case_id"].astype(str).tolist()
    val_case_ids = pt_df.iloc[val_idx]["case_id"].astype(str).tolist()

    train_labels = pt_df.iloc[train_idx]["y"].tolist()
    val_labels = pt_df.iloc[val_idx]["y"].tolist()

    # Molecular 分布
    if "Molecular" in pt_df.columns:
        train_molecular = pt_df.iloc[train_idx]["Molecular"].fillna("Unknown").astype(str).tolist()
        val_molecular = pt_df.iloc[val_idx]["Molecular"].fillna("Unknown").astype(str).tolist()
        val_mol_dist = dict(Counter(val_molecular))
    else:
        val_mol_dist = {}

    record = {
        "fold": int(fold_idx),
        "train_case_ids": train_case_ids,
        "val_case_ids": val_case_ids,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_label_pos": int(sum(train_labels)),
        "train_label_neg": int(len(train_labels) - sum(train_labels)),
        "val_label_pos": int(sum(val_labels)),
        "val_label_neg": int(len(val_labels) - sum(val_labels)),
        "val_label": dict(Counter(val_labels)),
        "val_molecular": val_mol_dist,
        "split_type": split_type,
    }

    if subgroup_name:
        record["subgroup"] = subgroup_name

    return record


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="基于总体划分提取各亚组划分，确保划分一致"
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="Molecular_label",
        choices=["Molecular_label", "Molecular", "label", "none"]
    )
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument(
        "--subgroups",
        type=str,
        nargs="+",
        default=["TNBC", "HER2", "HR+HER2-", "HR+HER2+"],
        help="要提取的亚组列表"
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="亚组最小样本数，低于此值跳过"
    )

    args = parser.parse_args()
    set_seed(args.seed)

    # 读取数据
    df = read_csv_smart(args.csv_path)
    pt_df, clinical_cols = build_patient_table(df, label_col=args.label_col)

    print(f"\n{'=' * 60}")
    print("📊 数据概况")
    print(f"{'=' * 60}")
    print(f"患者数: {len(pt_df)}")
    print(f"pCR(1): {int((pt_df['y'] == 1).sum())}")
    print(f"N-pCR(0): {int((pt_df['y'] == 0).sum())}")
    if "Molecular" in pt_df.columns:
        print("Molecular 分布:")
        for mol, count in pt_df["Molecular"].value_counts().items():
            print(f"  {mol}: {count}")
    print(f"{'=' * 60}\n")

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # ========== 1. 生成总体划分 ==========
    print("📊 生成总体划分...")

    n = len(pt_df)
    y = pt_df["y"].values

    # 构建分层标签
    if args.stratify_by == "Molecular_label":
        molecular = pt_df.get("Molecular", pd.Series(["Unknown"] * n)).fillna("Unknown").astype(str)
        stratify_labels = [f"{m}|y{int(l)}" for m, l in zip(molecular, y)]
        stratify_array = np.array(stratify_labels)
    elif args.stratify_by == "Molecular":
        stratify_array = pt_df.get("Molecular", pd.Series(["Unknown"] * n)).fillna("Unknown").astype(str).values
    elif args.stratify_by == "label":
        stratify_array = y
    else:
        stratify_array = None

    # 执行分层划分
    if stratify_array is not None:
        try:
            skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
            overall_folds = list(skf.split(np.arange(n), stratify_array))
        except ValueError as e:
            print(f"⚠️ 分层划分失败: {e}，回退到普通 KFold")
            kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
            overall_folds = list(kf.split(np.arange(n)))
    else:
        kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
        overall_folds = list(kf.split(np.arange(n)))

    # 保存总体划分
    overall_records = [
        build_fold_record(pt_df, fold, tr, va, split_type="overall")
        for fold, (tr, va) in enumerate(overall_folds)
    ]

    overall_meta = {
        "k": args.k,
        "seed": args.seed,
        "stratify_by": args.stratify_by,
        "n_samples": n,
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "split_type": "overall",
        "csv_path": os.path.abspath(args.csv_path),
        "label_col": args.label_col,
    }

    save_fold_splits(args.out_dir, overall_meta, overall_records)

    print("总体划分摘要:")
    for rec in overall_records:
        print(
            f"  Fold {rec['fold']}: train={rec['n_train']}, val={rec['n_val']}, "
            f"val_pos={rec['val_label_pos']}, val_neg={rec['val_label_neg']}"
        )

    # ========== 2. 从总体划分中提取亚组划分 ==========
    print(f"\n{'=' * 60}")
    print("📊 从总体划分中提取亚组划分...")
    print(f"{'=' * 60}")

    subgroups_dir = os.path.join(args.out_dir, "molecular_subgroups")
    os.makedirs(subgroups_dir, exist_ok=True)

    subgroup_summary = {}

    for subgroup_name in args.subgroups:
        print(f"\n--- 亚组: {subgroup_name} ---")

        # 从总体划分中提取亚组
        subgroup_folds, n_sub = extract_subgroup_folds(pt_df, overall_folds, subgroup_name)

        if n_sub < args.min_samples:
            print(f"⚠️ 样本数 ({n_sub}) < 最小要求 ({args.min_samples}), 跳过")
            continue

        # 检查每个 fold 是否有足够的样本
        valid = True
        for tr, va in subgroup_folds:
            if len(tr) < 2 or len(va) < 2:
                print(f"⚠️ 某个 fold 样本太少 (train={len(tr)}, val={len(va)}), 跳过")
                valid = False
                break

        if not valid:
            continue

        # 构建亚组划分记录
        sub_records = [
            build_fold_record(
                pt_df, fold, tr, va,
                split_type="subgroup",
                subgroup_name=subgroup_name
            )
            for fold, (tr, va) in enumerate(subgroup_folds)
        ]

        # 检查标签分布
        pos_counts = [r['val_label_pos'] for r in sub_records]
        neg_counts = [r['val_label_neg'] for r in sub_records]
        if min(pos_counts) < 1 or min(neg_counts) < 1:
            print(f"⚠️ 验证集标签不平衡: pos={min(pos_counts)}, neg={min(neg_counts)}, 跳过")
            continue

        # 保存亚组划分
        sub_out_dir = os.path.join(subgroups_dir, subgroup_name)
        os.makedirs(sub_out_dir, exist_ok=True)

        sub_meta = {
            "k": args.k,
            "seed": args.seed,
            "stratify_by": args.stratify_by,
            "n_samples": n_sub,
            "n_pos": int(
                (pt_df[pt_df.index.isin([i for i, _ in enumerate(pt_df)])].iloc[list(range(n_sub))] if False else 0)),
            # 简化
            "split_type": "subgroup",
            "subgroup": subgroup_name,
            "csv_path": os.path.abspath(args.csv_path),
            "label_col": args.label_col,
            "parent_split": os.path.basename(args.out_dir),
        }

        # 计算亚组标签分布
        subgroup_indices = []
        mapping = get_subgroup_molecular_map()
        for idx, row in pt_df.iterrows():
            if str(row.get("Molecular", "")).strip() in mapping.get(subgroup_name, []):
                subgroup_indices.append(idx)
        if subgroup_indices:
            sub_labels = [pt_df.iloc[i]["y"] for i in subgroup_indices]
            sub_meta["n_pos"] = int(sum(sub_labels))
            sub_meta["n_neg"] = int(len(sub_labels) - sum(sub_labels))

        save_fold_splits(sub_out_dir, sub_meta, sub_records)

        subgroup_summary[subgroup_name] = {
            "n_samples": n_sub,
            "folds": [{"train": r["n_train"], "val": r["n_val"],
                       "val_pos": r["val_label_pos"], "val_neg": r["val_label_neg"]}
                      for r in sub_records]
        }

        print(f"  ✅ {subgroup_name} 提取完成 (n={n_sub})")
        for r in sub_records:
            print(f"     Fold {r['fold']}: train={r['n_train']}, val={r['n_val']}, "
                  f"val_pos={r['val_label_pos']}, val_neg={r['val_label_neg']}")

    # ========== 保存亚组汇总 ==========
    if subgroup_summary:
        summary_path = os.path.join(subgroups_dir, "subgroup_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(subgroup_summary, f, indent=2)
        print(f"\n✅ 亚组汇总已保存: {summary_path}")

    # ========== 完成 ==========
    print(f"\n{'=' * 60}")
    print("✅ 所有划分完成！")
    print(f"{'=' * 60}")
    print(f"总体划分: {os.path.abspath(args.out_dir)}/kfold_splits.yaml")
    print(f"亚组划分: {os.path.abspath(args.out_dir)}/molecular_subgroups/")
    print(f"\n⚠️ 重要: 总体和亚组的 fold 划分完全一致，可公平比较！")


if __name__ == "__main__":
    main()



# 生成总体划分 + 所有亚组划分
# python make_kfold_splits2.py --csv_path Example_Dataset_Csv.csv --out_dir ./splits2/k5_stratify_molecular --k 5 --seed 1 --stratify_by Molecular_label







