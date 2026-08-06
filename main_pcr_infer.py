#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_pcr_infer.py
=================
乳腺 NAC pCR 二分类推理脚本。

输入：
  * --log_dir   : 训练日志目录（含 config.yaml 与 fold_*/checkpoint_*.pt，或 all_train 根目录权重）
  * --csv_path  : 与 example_dataset.csv 同格式的输入表
  * --save_dir  : 推理结果保存目录

输出：
  * fold_*/predictions.csv + metrics（逐折）
  * predictions_ensemble.csv + ensemble 指标（多折概率均值）
  * 若日志内有 kfold_splits.yaml 且 case 可对齐 → predictions_oof.csv + OOF 指标
  * infer_summary.yaml/json

示例：
  python main_pcr_infer.py \\
      --log_dir ../logs/abmil_pathology_e2 \\
      --csv_path example_dataset.csv \\
      --save_dir ../infer/abmil_pathology_e2
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# 与本脚本同目录的训练脚本
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import main_pcr_train as T  # noqa: E402


def discover_checkpoints(log_dir, preferred_name="checkpoint_best.pt"):
    """
    发现可用权重：
      * kfold: fold_*/checkpoint_*.pt → [(fold_idx, fold_dir, ckpt_path), ...]
      * all_train: 日志根目录下 checkpoint_*.pt → [(-1, log_dir, ckpt_path)]
    """
    log_dir = os.path.abspath(log_dir)
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"日志目录不存在: {log_dir}")

    items = []
    for name in sorted(os.listdir(log_dir)):
        if not name.startswith("fold_"):
            continue
        fold_dir = os.path.join(log_dir, name)
        if not os.path.isdir(fold_dir):
            continue
        try:
            fold_idx = int(name.split("_", 1)[1])
        except Exception:
            continue
        ckpt = T.resolve_ckpt_path(
            fold_dir, prefer_best=True, preferred_name=preferred_name
        )
        if ckpt is None:
            print(f"警告: {fold_dir} 未找到 checkpoint，跳过")
            continue
        items.append((fold_idx, fold_dir, ckpt))
    items.sort(key=lambda x: x[0])

    if items:
        return items, "kfold"

    # all_train：权重在日志根目录
    ckpt = T.resolve_ckpt_path(
        log_dir, prefer_best=True, preferred_name=preferred_name
    )
    if ckpt is None:
        raise FileNotFoundError(
            f"在 {log_dir} 下未找到 fold_*/checkpoint_*.pt，"
            f"也未找到根目录 checkpoint_best/last.pt"
        )
    return [(-1, log_dir, ckpt)], "all_train"


def load_train_config(log_dir, config_override=None):
    """从训练日志加载超参；可用 --config 覆盖。"""
    log_dir = os.path.abspath(log_dir)
    cfg = {}
    cfg_path = None
    for name in ("config.yaml", "config.yml", "config.json"):
        cand = os.path.join(log_dir, name)
        if os.path.isfile(cand):
            cfg_path = cand
            break
    if cfg_path is None:
        raise FileNotFoundError(
            f"日志目录缺少 config.yaml/json: {log_dir}"
        )
    cfg.update(T.load_config_file(cfg_path))
    print(f"已加载训练配置: {cfg_path}")

    if config_override and os.path.isfile(config_override):
        override = T.load_config_file(config_override)
        for k, v in override.items():
            if k in T.HPARAM_KEYS:
                cfg[k] = v
        print(f"已用 {config_override} 覆盖部分超参")
    return cfg


def load_encoder_for_dir(model_dir, cfg):
    """按需加载 clinical_encoder.json；pathology 模式返回 None。"""
    modality = T.normalize_modality(cfg)
    enc_path = os.path.join(model_dir, "clinical_encoder.json")
    if modality in ("pathomic", "clinical"):
        if os.path.isfile(enc_path):
            return T.load_clinical_encoder(enc_path)
        if modality == "clinical":
            raise FileNotFoundError(f"clinical 推理需要 {enc_path}")
        print(f"警告: 未找到 {enc_path}，该模型将不使用临床特征")
        return None
    return None


def maybe_disable_clinical(cfg, encoder):
    """需要临床但无 encoder 时降级为 pathology。"""
    modality = T.normalize_modality(cfg)
    if encoder is not None:
        cfg = dict(cfg)
        cfg["clinical_in_dim"] = int(encoder.output_dim)
        return cfg, encoder
    if modality == "clinical":
        raise FileNotFoundError("clinical 推理需要 clinical_encoder.json")
    if modality == "pathomic":
        print("警告: 未找到 clinical_encoder，降级为 pathology 推理")
        cfg = dict(cfg)
        cfg["modality"] = "pathology"
        cfg["use_clinical"] = False
        cfg["clinical_only"] = False
        cfg["clinical_in_dim"] = 0
    return cfg, None


def prepare_patient_table(cfg, csv_path):
    """读取 CSV → 患者表；必要时自动推断 in_dim。"""
    modality = T.normalize_modality(cfg)
    df = T.read_csv_smart(csv_path)
    pt, clinical_cols, feat_col = T.build_patient_table(
        df,
        label_col=cfg.get("label_col", "label"),
        feat_path_col=cfg.get("feat_path_col"),
        require_feats=(modality != "clinical"),
    )
    if feat_col is not None:
        cfg["feat_path_col"] = feat_col
    if modality == "clinical":
        cfg["in_dim"] = 0
    elif cfg.get("in_dim") is None or int(cfg.get("in_dim") or 0) <= 0:
        all_paths = [p for ps in pt["feat_paths"] for p in ps]
        cfg["in_dim"] = T.detect_in_dim(all_paths, cfg.get("feat_key", "features"))
    print(
        f"推理样本: n={len(pt)}, modality={modality}, in_dim={cfg.get('in_dim')}, "
        f"临床列={clinical_cols}"
    )
    return pt


def ensemble_predictions(fold_pred_map):
    """按 case_id 对多折 prob_pCR 取均值。"""
    merged = None
    for fold_idx, pred_df in sorted(fold_pred_map.items()):
        tmp = pred_df[["case_id", "label", "Molecular", "prob_pCR"]].rename(
            columns={"prob_pCR": f"prob_pCR_fold{fold_idx}"}
        )
        if merged is None:
            merged = tmp
        else:
            merged = merged.merge(
                tmp.drop(columns=["label", "Molecular"], errors="ignore"),
                on="case_id", how="outer",
            )
    prob_cols = [c for c in merged.columns if c.startswith("prob_pCR_fold")]
    ens = merged[["case_id", "label", "Molecular"]].copy()
    ens["prob_pCR"] = merged[prob_cols].astype(float).mean(axis=1)
    ens["pred"] = (ens["prob_pCR"] >= 0.5).astype(int)
    ens["n_folds"] = merged[prob_cols].notna().sum(axis=1).astype(int)
    for c in prob_cols:
        ens[c] = merged[c]
    return ens


def try_oof_predictions(log_dir, pt, fold_pred_map, save_dir, cfg=None):
    """若日志内有划分文件且 case 可对齐，生成 OOF 预测与指标。"""
    splits_file = None
    for name in ("kfold_splits.yaml", "kfold_splits.yml", "kfold_splits.json"):
        cand = os.path.join(log_dir, name)
        if os.path.isfile(cand):
            splits_file = cand
            break
    if splits_file is None:
        print("未找到日志内 kfold_splits.*，跳过 OOF")
        return None

    try:
        splits, split_meta, _ = T.load_kfold_splits(splits_file, pt)
    except Exception as e:
        print(f"警告: 无法加载划分做 OOF（将跳过）: {e}")
        return None

    oof_rows = []
    for fold_idx, (_tr, va_idx) in enumerate(splits):
        if fold_idx not in fold_pred_map:
            continue
        va_ids = set(pt.iloc[va_idx]["case_id"].astype(str).tolist())
        sub = fold_pred_map[fold_idx]
        sub = sub[sub["case_id"].astype(str).isin(va_ids)].copy()
        oof_rows.append(sub)
    if not oof_rows:
        print("警告: OOF 无有效样本，跳过")
        return None

    oof_df = pd.concat(oof_rows, ignore_index=True)
    _, oof_metrics = T.save_prediction_outputs(
        save_dir, oof_df, prefix="oof",
        extra_meta={
            "log_dir": os.path.abspath(log_dir),
            "splits_path": splits_file,
            "mode": "oof",
            "subgroup": split_meta.get("subgroup"),
            "split_type": split_meta.get("split_type"),
            "label_map": {"N-pCR": 0, "pCR": 1},
        },
        cfg=cfg,
    )
    oof_df.to_csv(os.path.join(save_dir, "predictions_oof.csv"), index=False)
    T.print_metrics_block("[OOF]", oof_metrics)
    return oof_metrics


def run_infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    log_dir = os.path.abspath(args.log_dir)
    csv_path = os.path.abspath(args.csv_path)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    cfg = load_train_config(log_dir, config_override=args.config)
    if getattr(args, "n_boot", None) is not None:
        cfg["n_boot"] = int(args.n_boot)
    if getattr(args, "bootstrap_ci", None) is not None:
        cfg["bootstrap_ci"] = float(args.bootstrap_ci)
    T.set_seed(int(cfg.get("seed", 1)))
    modality = T.normalize_modality(cfg)
    pt = prepare_patient_table(cfg, csv_path)

    ckpt_items, mode = discover_checkpoints(log_dir, preferred_name=args.ckpt_name)
    print(f"发现 {len(ckpt_items)} 个权重（mode={mode}，优先 {args.ckpt_name}）")

    fold_pred_map = {}
    fold_records = []

    for fold_idx, fold_dir, ckpt_path in ckpt_items:
        tag = f"fold_{fold_idx}" if fold_idx >= 0 else "all_train"
        print(f"\n----- Infer {tag}: {ckpt_path} -----")
        fold_cfg = dict(cfg)
        encoder = load_encoder_for_dir(fold_dir, fold_cfg)
        fold_cfg, encoder = maybe_disable_clinical(fold_cfg, encoder)
        model, fold_cfg = T.load_model_for_eval(
            fold_cfg, device, ckpt_path, encoder=encoder
        )
        pred_df = T.predict_patient_table(
            model, pt, fold_cfg, device, clinical_encoder=encoder
        )
        if fold_idx >= 0:
            pred_df["fold"] = fold_idx
            fold_pred_map[fold_idx] = pred_df
            fold_out = os.path.join(save_dir, f"fold_{fold_idx}")
        else:
            fold_pred_map[0] = pred_df
            fold_out = save_dir

        _, fold_metrics = T.save_prediction_outputs(
            fold_out, pred_df, prefix="infer",
            extra_meta={
                "ckpt_path": os.path.abspath(ckpt_path),
                "csv_path": csv_path,
                "log_dir": log_dir,
                "fold": fold_idx if fold_idx >= 0 else None,
                "label_map": {"N-pCR": 0, "pCR": 1},
            },
            cfg=cfg,
        )
        pred_df.to_csv(os.path.join(fold_out, "predictions.csv"), index=False)
        T.print_metrics_block(f"[{tag}]", fold_metrics)
        fold_records.append({
            "fold": fold_idx if fold_idx >= 0 else None,
            "ckpt_path": os.path.abspath(ckpt_path),
            "fold_dir": fold_dir,
            "metrics": fold_metrics,
        })

    ens_metrics = None
    mean_fold_metrics = None
    if mode == "kfold" and len(fold_pred_map) > 1:
        ens = ensemble_predictions(fold_pred_map)
        _, ens_metrics = T.save_prediction_outputs(
            save_dir, ens, prefix="ensemble",
            extra_meta={
                "log_dir": log_dir,
                "csv_path": csv_path,
                "n_models": int(ens["n_folds"].max()) if len(ens) else 0,
                "mode": "ensemble_mean",
                "label_map": {"N-pCR": 0, "pCR": 1},
            },
            cfg=cfg,
        )
        ens.to_csv(os.path.join(save_dir, "predictions_ensemble.csv"), index=False)
        T.print_metrics_block("[Ensemble]", ens_metrics)

        # 各折指标的折级 bootstrap 平均
        n_boot, boot_seed, boot_ci = T.resolve_bootstrap_params(cfg=cfg)
        mean_fold_metrics = T.aggregate_fold_metrics_bootstrap(
            [r["metrics"] for r in fold_records],
            n_boot=n_boot, seed=boot_seed, ci=boot_ci,
        )
        with open(os.path.join(save_dir, "mean_fold_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(T._to_builtin(mean_fold_metrics), f, ensure_ascii=False, indent=2)
        T.save_yaml(mean_fold_metrics, os.path.join(save_dir, "mean_fold_metrics.yaml"))
        T.print_metrics_block("[K折平均(折级bootstrap)]", mean_fold_metrics)
    elif len(fold_pred_map) == 1:
        # 单模型：根目录再落一份简名结果
        only = next(iter(fold_pred_map.values()))
        only.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
        ens_metrics = fold_records[0]["metrics"]

    oof_metrics = None
    if mode == "kfold":
        oof_metrics = try_oof_predictions(log_dir, pt, fold_pred_map, save_dir, cfg=cfg)

    summary = {
        "log_dir": log_dir,
        "csv_path": csv_path,
        "save_dir": save_dir,
        "modality": modality,
        "model_type": cfg.get("model_type"),
        "mode": mode,
        "n_models": len(ckpt_items),
        "n_patients": int(len(pt)),
        "ensemble_metrics": ens_metrics,
        "mean_fold_metrics": mean_fold_metrics,
        "oof_metrics": oof_metrics,
        "n_boot": cfg.get("n_boot"),
        "bootstrap_ci": cfg.get("bootstrap_ci"),
        "folds": [
            {
                "fold": r["fold"],
                "ckpt_path": r["ckpt_path"],
                "fold_dir": r["fold_dir"],
            }
            for r in fold_records
        ],
        "label_map": {"N-pCR": 0, "pCR": 1},
    }
    T.save_yaml(summary, os.path.join(save_dir, "infer_summary.yaml"))
    with open(os.path.join(save_dir, "infer_summary.json"), "w", encoding="utf-8") as f:
        json.dump(T._to_builtin(summary), f, ensure_ascii=False, indent=2)

    print(f"\n推理完成，结果目录: {save_dir}")
    print(f"  - 逐折: {save_dir}/fold_*/predictions.csv")
    if mode == "kfold" and len(fold_pred_map) > 1:
        print(f"  - Ensemble: {save_dir}/predictions_ensemble.csv")
    if oof_metrics is not None:
        print(f"  - OOF: {save_dir}/predictions_oof.csv")
    print(f"  - 摘要: {save_dir}/infer_summary.yaml")
    return summary


def get_args():
    p = argparse.ArgumentParser(description="乳腺 NAC pCR 二分类推理脚本")
    p.add_argument(
        "--log_dir", type=str, required=True,
        help="训练日志目录（含 config.yaml 与 fold_*/checkpoint_*.pt）",
    )
    p.add_argument(
        "--csv_path", type=str, required=True,
        help="输入 CSV（格式同 example_dataset.csv）",
    )
    p.add_argument(
        "--save_dir", type=str, required=True,
        help="推理结果保存目录",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="可选：额外超参 yaml/json，覆盖日志内 config 的 HPARAM 字段",
    )
    p.add_argument(
        "--ckpt_name", type=str, default="checkpoint_best.pt",
        help="各 fold 优先使用的权重文件名（不存在则回退 last/best_loss）",
    )
    p.add_argument(
        "--n_boot", type=int, default=None,
        help="bootstrap 次数（默认读取训练 config 的 n_boot，否则 1000；<=0 关闭）",
    )
    p.add_argument(
        "--bootstrap_ci", type=float, default=None,
        help="bootstrap 置信水平（默认读取训练 config，否则 0.95）",
    )
    return p.parse_args()


def main():
    args = get_args()
    run_infer(args)


if __name__ == "__main__":
    main()
