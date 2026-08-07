#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_kfold_splits_and_test.py
=============================
先按比例划出独立 test 集，再对剩余样本做 K 折划分。
K 折逻辑与 make_kfold_splits.py 完全一致（总体 + 亚组提取）。

输出目录结构：
  {out_dir}/
    kfold_splits.yaml              # 总体划分（仅 train/val，不含 test）
    split_config.yaml
    test_case_ids.txt              # 独立 test 患者 ID
    test_dataset.csv               # 独立 test，example_dataset.csv 格式（供 infer）
    fold_0/...
    molecular_subgroups/
      TNBC/
        kfold_splits.yaml
        fold_0/...
      ...

示例：
  python make_kfold_splits_and_test.py \\
      --csv_path example_dataset.csv \\
      --out_dir ./splits/k5_mol_label_test0.2 \\
      --k 5 --seed 1 --test_ratio 0.2 --stratify_by Molecular_label
"""

from __future__ import print_function

import argparse
import os
import json
import yaml
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


# ============================================================================
# 工具函数（与 make_kfold_splits.py 保持一致）
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


def build_stratify_array(pt_df, stratify_by):
    """构建分层标签，与 make_kfold_splits.py 一致"""
    n = len(pt_df)
    y = pt_df["y"].values
    if stratify_by == "Molecular_label":
        molecular = pt_df.get("Molecular", pd.Series(["Unknown"] * n)).fillna("Unknown").astype(str)
        stratify_labels = [f"{m}|y{int(l)}" for m, l in zip(molecular, y)]
        return np.array(stratify_labels)
    elif stratify_by == "Molecular":
        return pt_df.get("Molecular", pd.Series(["Unknown"] * n)).fillna("Unknown").astype(str).values
    elif stratify_by == "label":
        return y
    else:
        return None


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

    subgroup_indices = []
    for idx, row in pt_df.iterrows():
        mol = str(row.get("Molecular", "")).strip()
        if mol in molecular_values:
            subgroup_indices.append(idx)

    subgroup_indices_set = set(subgroup_indices)

    subgroup_folds = []
    for train_idx, val_idx in overall_folds:
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


def save_test_dataset(df, test_case_ids, out_path):
    """按 example_dataset.csv 格式保存独立 test 集（slide 级，保留原始列）"""
    test_ids = set(str(x) for x in test_case_ids)
    test_df = df[df["case_id"].astype(str).isin(test_ids)].copy()
    # 保持原始列顺序与行顺序
    test_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return test_df


def hold_out_test(pt_df, test_ratio, seed, stratify_by):
    """先划出独立 test，返回 (dev_df, test_case_ids, test_info)"""
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"--test_ratio 须在 (0, 1) 内，得到: {test_ratio}")

    n = len(pt_df)
    indices = np.arange(n)
    stratify_array = build_stratify_array(pt_df, stratify_by)

    try:
        if stratify_array is not None:
            dev_idx, test_idx = train_test_split(
                indices,
                test_size=test_ratio,
                random_state=seed,
                shuffle=True,
                stratify=stratify_array,
            )
        else:
            dev_idx, test_idx = train_test_split(
                indices,
                test_size=test_ratio,
                random_state=seed,
                shuffle=True,
            )
    except ValueError as e:
        print(f"⚠️ test 分层划分失败: {e}，回退到非分层划分")
        dev_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            shuffle=True,
        )

    test_case_ids = pt_df.iloc[test_idx]["case_id"].astype(str).tolist()
    test_labels = pt_df.iloc[test_idx]["y"].tolist()
    test_info = {
        "n_test": int(len(test_idx)),
        "n_test_pos": int(sum(test_labels)),
        "n_test_neg": int(len(test_labels) - sum(test_labels)),
        "test_case_ids": test_case_ids,
    }
    if "Molecular" in pt_df.columns:
        test_mol = pt_df.iloc[test_idx]["Molecular"].fillna("Unknown").astype(str).tolist()
        test_info["test_molecular"] = dict(Counter(test_mol))

    # 剩余样本重置 index，供后续 K 折使用
    dev_df = pt_df.iloc[dev_idx].reset_index(drop=True)
    return dev_df, test_case_ids, test_info


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="先划出独立 test，再对剩余样本做 K 折（逻辑同 make_kfold_splits.py）"
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="独立 test 集比例（患者级），先划出后再对剩余做 K 折",
    )
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="Molecular_label",
        choices=["Molecular_label", "Molecular", "label", "none"],
        help="test 划分与 K 折划分共用同一分层策略",
    )
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument(
        "--subgroups",
        type=str,
        nargs="+",
        default=["TNBC", "HER2", "HR+HER2-", "HR+HER2+"],
        help="要提取的亚组列表",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="亚组最小样本数，低于此值跳过",
    )
    parser.add_argument(
        "--test_csv_name",
        type=str,
        default="test_dataset.csv",
        help="独立 test CSV 文件名（example_dataset.csv 格式）",
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

    os.makedirs(args.out_dir, exist_ok=True)

    # ========== 0. 先划出独立 test ==========
    print(f"📊 划出独立 test (ratio={args.test_ratio}, stratify_by={args.stratify_by})...")
    dev_df, test_case_ids, test_info = hold_out_test(
        pt_df, args.test_ratio, args.seed, args.stratify_by
    )

    test_ids_path = os.path.join(args.out_dir, "test_case_ids.txt")
    with open(test_ids_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_case_ids))

    test_csv_path = os.path.join(args.out_dir, args.test_csv_name)
    test_df = save_test_dataset(df, test_case_ids, test_csv_path)

    print(
        f"  test: n={test_info['n_test']} "
        f"(pos={test_info['n_test_pos']}, neg={test_info['n_test_neg']}), "
        f"slides={len(test_df)}"
    )
    print(f"  ✅ test case IDs: {test_ids_path}")
    print(f"  ✅ test CSV (infer 格式): {test_csv_path}")

    # ========== 1. 对剩余样本做总体 K 折 ==========
    print(f"\n📊 对剩余样本生成总体 K 折划分 (n={len(dev_df)})...")

    pt_df = dev_df  # 后续 K 折与亚组提取均基于剩余样本
    n = len(pt_df)
    y = pt_df["y"].values
    stratify_array = build_stratify_array(pt_df, args.stratify_by)

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

    overall_records = [
        build_fold_record(pt_df, fold, tr, va, split_type="overall")
        for fold, (tr, va) in enumerate(overall_folds)
    ]

    overall_meta = {
        "k": args.k,
        "seed": args.seed,
        "stratify_by": args.stratify_by,
        "test_ratio": args.test_ratio,
        "n_samples": n,
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "n_test": test_info["n_test"],
        "n_test_pos": test_info["n_test_pos"],
        "n_test_neg": test_info["n_test_neg"],
        "test_case_ids": test_case_ids,
        "test_csv_path": os.path.abspath(test_csv_path),
        "split_type": "overall_with_heldout_test",
        "csv_path": os.path.abspath(args.csv_path),
        "label_col": args.label_col,
    }

    save_fold_splits(args.out_dir, overall_meta, overall_records)

    # 额外保存 split_config
    split_config = {
        "k": args.k,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "stratify_by": args.stratify_by,
        "csv_path": os.path.abspath(args.csv_path),
        "test_csv_path": os.path.abspath(test_csv_path),
        "n_total": int(n + test_info["n_test"]),
        "n_dev": n,
        "n_test": test_info["n_test"],
        "n_test_pos": test_info["n_test_pos"],
        "n_test_neg": test_info["n_test_neg"],
    }
    config_path = os.path.join(args.out_dir, "split_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(split_config, f, default_flow_style=False, allow_unicode=True)

    print("总体划分摘要 (不含 test):")
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

        subgroup_folds, n_sub = extract_subgroup_folds(pt_df, overall_folds, subgroup_name)

        if n_sub < args.min_samples:
            print(f"⚠️ 样本数 ({n_sub}) < 最小要求 ({args.min_samples}), 跳过")
            continue

        valid = True
        for tr, va in subgroup_folds:
            if len(tr) < 2 or len(va) < 2:
                print(f"⚠️ 某个 fold 样本太少 (train={len(tr)}, val={len(va)}), 跳过")
                valid = False
                break

        if not valid:
            continue

        sub_records = [
            build_fold_record(
                pt_df, fold, tr, va,
                split_type="subgroup",
                subgroup_name=subgroup_name
            )
            for fold, (tr, va) in enumerate(subgroup_folds)
        ]

        pos_counts = [r['val_label_pos'] for r in sub_records]
        neg_counts = [r['val_label_neg'] for r in sub_records]
        if min(pos_counts) < 1 or min(neg_counts) < 1:
            print(f"⚠️ 验证集标签不平衡: pos={min(pos_counts)}, neg={min(neg_counts)}, 跳过")
            continue

        sub_out_dir = os.path.join(subgroups_dir, subgroup_name)
        os.makedirs(sub_out_dir, exist_ok=True)

        sub_meta = {
            "k": args.k,
            "seed": args.seed,
            "stratify_by": args.stratify_by,
            "test_ratio": args.test_ratio,
            "n_samples": n_sub,
            "n_pos": 0,
            "n_neg": 0,
            "split_type": "subgroup",
            "subgroup": subgroup_name,
            "csv_path": os.path.abspath(args.csv_path),
            "label_col": args.label_col,
            "parent_split": os.path.basename(args.out_dir),
            "test_csv_path": os.path.abspath(test_csv_path),
        }

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

    if subgroup_summary:
        summary_path = os.path.join(subgroups_dir, "subgroup_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(subgroup_summary, f, indent=2)
        print(f"\n✅ 亚组汇总已保存: {summary_path}")

    print(f"\n{'=' * 60}")
    print("✅ 所有划分完成！")
    print(f"{'=' * 60}")
    print(f"独立 test CSV: {os.path.abspath(test_csv_path)}")
    print(f"总体划分: {os.path.abspath(args.out_dir)}/kfold_splits.yaml")
    print(f"亚组划分: {os.path.abspath(args.out_dir)}/molecular_subgroups/")
    print(f"\n⚠️ 重要: test 已先划出；K 折仅在剩余样本上，总体与亚组 fold 一致。")
    print(f"推理示例:")
    print(f"  python main_pcr_infer.py --log_dir <训练日志> --csv_path {test_csv_path} --save_dir <输出>")


if __name__ == "__main__":
    main()


# 示例：
# python make_kfold_splits_and_test.py \
#     --csv_path example_dataset.csv \
#     --out_dir ./splits/k5_mol_label_test0.2 \
#     --k 5 --seed 1 --test_ratio 0.2 --stratify_by Molecular_label
