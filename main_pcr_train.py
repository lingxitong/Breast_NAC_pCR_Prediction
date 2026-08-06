#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_pcr.py
===========
基于 BreastRCB-Prognosis 中期融合框架改造的乳腺新辅助治疗 pCR 二分类训练脚本。

任务：
  * 标签映射：N-pCR -> 0，pCR -> 1
  * 损失：CrossEntropy
  * 评价：AUC / AUPRC / Accuracy / Balanced Acc / F1 / Precision / Recall
    （Sensitivity）/ Specificity / PPV / NPV / MCC，以及 TP/TN/FP/FN
  * 终末评估（单折/OOF/ensemble 等）对样本做 bootstrap×1000 给出 95% CI；
    K 折平均时对各折指标再做 bootstrap×1000 给出均值的 95% CI
  * 一个患者(case)可能有多张 slide，训练时拼接成一个 bag；若 slide 数 > max_slides
    则随机选取 max_slides 张拼接；验证时拼接全部 slide。

输入 CSV 格式见项目根目录 example_dataset.csv，关键列：
  case_id, slide_id, slide_feats_path, label,
  Molecular, T, N, Age, ER, PR, HER2, Ki67
其中 label 可为字符串 N-pCR/pCR，或已映射的 0/1。
slide_feats_path 指向每张 slide 的特征文件（.pt 或 .h5）。
临床融合白名单列：
  因子变量（one-hot）：Molecular, T, N, HER2
  连续变量（标准化）：Age, ER, PR, Ki67
Molecular 取值（四种）：HR+HER2- / HR+HER2+ / TNBC / HER2。

训练：
  * --split_mode 控制 kfold（默认）或 all_train。
  * 结束后保存验证集逐样本概率，并按 Molecular 输出总体/亚组指标。
  * 早停（有验证集时）：--early_stop/--no-early_stop、--patience、
    --min_delta、--early_stop_metric（默认 val_auc）。

模态（--modality）:
  pathomic  : WSI MIL + 临床中期融合（默认）
  pathology : 仅病理 WSI
  clinical  : 仅临床信息（不加载 slide 特征）；也可用 --clinical_only

K 折划分：
  * 必须先用独立脚本 make_kfold_splits.py 生成划分；
  * 训练强制依赖 --splits_path（预划分结果）；CSV 路径从划分文件
    meta.csv_path 读取，不再单独要求 --csv_path。
  * main_pcr.py 内不再做现场/临时划分。
  * 亚组专训（splits 来自 molecular_subgroups/<Molecular>/）时，若能定位到父级
    总体划分，会额外在总体验证集上评估，并输出总体指标 + 各 Molecular 亚组指标。

超参数：训练时以 config.yaml 写入日志目录；可用 --config 覆盖。

示例（在项目根目录执行）：
  # 1) 先划分（唯一划分入口；会把 csv_path 写入划分 meta）
  python make_kfold_splits.py --csv_path example_dataset.csv \\
      --out_dir ./splits/mol_label_k5 --k 5 --stratify_by Molecular_label

  # 2) 基于预划分训练（病理+临床）
  python main_pcr.py --split_mode kfold \\
      --splits_path ./splits/mol_label_k5/kfold_splits.yaml \\
      --log_root ./logs --exp_name pcr_kfold

  # 3) 仅临床信息训练
  python main_pcr.py --split_mode kfold --clinical_only \\
      --splits_path ./splits/mol_label_k5/kfold_splits.yaml \\
      --log_root ./logs --exp_name pcr_clinical_only
"""

from __future__ import print_function

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold

# 写入 metrics.csv / 日志的逐 epoch 指标键（run_epoch 返回值中的键名）
EPOCH_METRIC_KEYS = (
    "loss", "auc", "auprc", "acc", "balanced_acc",
    "f1", "precision", "recall", "sensitivity", "specificity",
    "ppv", "npv", "mcc", "tp", "tn", "fp", "fn",
)

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


# ----------------------------------------------------------------------------
# YAML / JSON 配置 IO
# ----------------------------------------------------------------------------
def _to_builtin(obj):
    """将 numpy / pandas 标量转为可 YAML/JSON 序列化的 Python 内置类型。"""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_builtin(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            _to_builtin(obj), f,
            allow_unicode=True, sort_keys=False, default_flow_style=False,
        )


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config_file(path):
    """加载超参数配置，支持 .yaml/.yml/.json。"""
    path = str(path)
    lower = path.lower()
    if lower.endswith((".yaml", ".yml")):
        data = load_yaml(path)
    elif lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # 扩展名不明时优先按 YAML，失败再尝试 JSON
        try:
            data = load_yaml(path)
        except Exception:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层应为 mapping/dict: {path}")
    return data

# ----------------------------------------------------------------------------
# 超参数 / 列约定
# ----------------------------------------------------------------------------
HPARAM_KEYS = [
    "k", "split_mode", "stratify_by", "splits_path", "csv_path", "n_classes",
    "max_epochs", "lr", "reg",
    "drop_out", "gc", "seed", "opt", "model_type", "in_dim",
    "hidden_dim", "max_slides_train", "feat_key", "num_workers",
    "early_stop", "patience", "min_delta", "early_stop_metric",
    "n_boot", "bootstrap_ci",
    "mambamil_layer", "mambamil_rate", "mambamil_type",
    "modality", "clinical_only", "use_clinical",
    "fusion_type", "clinical_hidden_dim", "clinical_in_dim",
    "label_col", "feat_path_col",
]

# 需要报告 bootstrap 95% CI 的性能指标（不含混淆矩阵计数）
BOOTSTRAP_METRIC_KEYS = (
    "auc", "auprc", "acc", "balanced_acc",
    "f1", "precision", "recall", "sensitivity", "specificity",
    "ppv", "npv", "mcc",
)
DEFAULT_N_BOOT = 1000
DEFAULT_BOOTSTRAP_CI = 0.95

# 早停监控指标
EARLY_STOP_METRIC_CHOICES = (
    "val_auc", "val_auprc", "val_loss", "val_acc", "val_balanced_acc",
    "val_f1", "val_precision", "val_recall", "val_sensitivity",
    "val_specificity", "val_mcc",
)

# K 折分层依据（写入 config.yaml / kfold_splits.yaml 的 stratify_by）
STRATIFY_BY_CHOICES = ("Molecular_label", "Molecular", "label", "none")
# 输入模态
MODALITY_CHOICES = ("pathomic", "pathology", "clinical")

# 标识 / 路径 / 标签列（禁止作为临床特征）
META_COLS = {
    "case_id", "slide_id",
    "slide_feats_path", "slide_feat_path",
    "label", "y", "feat_paths",
}
# 不再排除分子分型；保留空集以兼容旧逻辑
EXCLUDED_CLINICAL = set()
# 因子变量：one-hot 编码（Molecular / T / N / HER2）
CATEGORICAL_COLS = ["Molecular", "T", "N", "HER2"]
# 连续变量：z-score 标准化（Age / ER / PR / Ki67）
NUMERIC_COLS = ["Age", "ER", "PR", "Ki67"]
# 临床白名单：仅这些列参与融合（若 CSV 中存在）
CLINICAL_WHITELIST = CATEGORICAL_COLS + NUMERIC_COLS
KNOWN_CATEGORICAL = set(CATEGORICAL_COLS)
KNOWN_NUMERIC = set(NUMERIC_COLS)
# 预定义类别（训练时与数据中出现的取值取并集，保证折间维度稳定）
# 分子分型预置四类（即使某折未出现也保留 one-hot 维度）
MOLECULAR_CATEGORIES = ["HR+HER2-", "HR+HER2+", "TNBC", "HER2"]
KNOWN_CATEGORIES = {
    "Molecular": list(MOLECULAR_CATEGORIES),
}

LABEL_MAP = {
    "n-pcr": 0, "n_pcr": 0, "npcr": 0, "0": 0, 0: 0,
    "pcr": 1, "1": 1, 1: 1,
}


def map_label(v):
    """将 N-pCR/pCR（或 0/1）映射为二分类标签。"""
    if pd.isna(v):
        raise ValueError("label 存在缺失值")
    if isinstance(v, (int, np.integer)):
        key = int(v)
    else:
        key = str(v).strip()
        key_lower = key.lower().replace(" ", "")
        if key_lower in LABEL_MAP:
            return LABEL_MAP[key_lower]
        # 兼容 "N-pCR" 等大小写混合
        if key_lower.replace("-", "") in LABEL_MAP:
            return LABEL_MAP[key_lower.replace("-", "")]
        try:
            key = int(float(key))
        except Exception as e:
            raise ValueError(f"无法解析 label 值: {v!r}") from e
    if key not in (0, 1):
        raise ValueError(f"label 必须为 0/1 或 N-pCR/pCR，得到: {v!r}")
    return int(key)


def resolve_feat_path_col(df, preferred=None):
    if preferred and preferred in df.columns:
        return preferred
    for col in ("slide_feats_path", "slide_feat_path"):
        if col in df.columns:
            return col
    raise KeyError("CSV 缺少 slide_feats_path / slide_feat_path 列")


def get_clinical_columns(df):
    """仅返回白名单中且实际存在的临床列（因子 + 连续）。"""
    cols = []
    for c in CLINICAL_WHITELIST:
        if c in df.columns and c not in EXCLUDED_CLINICAL and c not in META_COLS:
            cols.append(c)
    return cols


# ============================================================================
# 模型：MIL 聚合器 -> 全局表征；可选与临床特征中期融合 -> 分类头
# ============================================================================
def _init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class ABMILBackbone(nn.Module):
    """Gated-Attention MIL，输出 slide 级全局表征 [1, hidden]。"""

    def __init__(self, in_dim, dropout=0.25, hidden=512, att_dim=256):
        super().__init__()
        self.hidden_dim = hidden
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden, att_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden, att_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(att_dim, 1)
        self.apply(_init_weights)

    def forward(self, x):
        h = self.fc(x)
        A = self.attention_w(self.attention_V(h) * self.attention_U(h))
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        return torch.mm(A, h)


class MeanMaxBackbone(nn.Module):
    def __init__(self, in_dim, dropout=0.25, hidden=512, pool="mean"):
        super().__init__()
        self.hidden_dim = hidden
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.pool = pool
        self.apply(_init_weights)

    def forward(self, x):
        h = self.fc(x)
        return h.mean(dim=0, keepdim=True) if self.pool == "mean" else h.max(dim=0, keepdim=True)[0]


class ConcatFusion(nn.Module):
    def __init__(self, mil_dim, clin_dim, out_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(mil_dim + clin_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, mil_repr, clin_repr):
        return self.net(torch.cat([mil_repr, clin_repr], dim=-1))


class BilinearFusion(nn.Module):
    def __init__(self, mil_dim, clin_dim, out_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(mil_dim, clin_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, mil_repr, clin_repr):
        return self.norm(F.relu(self.bilinear(mil_repr, clin_repr)))


class GatedFusion(nn.Module):
    """门控融合：根据 [全局表征; 临床嵌入] 学习逐维权重。"""

    def __init__(self, mil_dim, clin_dim, out_dim):
        super().__init__()
        self.proj_mil = nn.Linear(mil_dim, out_dim)
        self.proj_clin = nn.Linear(clin_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(mil_dim + clin_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, mil_repr, clin_repr):
        gate = self.gate(torch.cat([mil_repr, clin_repr], dim=-1))
        return gate * self.proj_mil(mil_repr) + (1.0 - gate) * self.proj_clin(clin_repr)


def build_fusion(fusion_type, mil_dim, clin_dim, out_dim, dropout=0.25):
    if fusion_type == "concat":
        return ConcatFusion(mil_dim, clin_dim, out_dim, dropout)
    if fusion_type == "bilinear":
        return BilinearFusion(mil_dim, clin_dim, out_dim)
    if fusion_type == "gated":
        return GatedFusion(mil_dim, clin_dim, out_dim)
    raise ValueError(f"未知 fusion_type: {fusion_type}，可选 concat/bilinear/gated")


class PathomicClassificationModel(nn.Module):
    """
    中期融合二分类模型：
      bag -> MIL backbone -> 全局表征
      临床向量 -> MLP -> 临床嵌入
      fusion(全局表征, 临床嵌入) -> 分类 logits [n_classes]
    """

    def __init__(self, backbone, n_classes, clinical_in_dim, fusion_type,
                 hidden_dim=512, dropout=0.25, use_clinical=True):
        super().__init__()
        self.backbone = backbone
        self.use_clinical = use_clinical and clinical_in_dim > 0
        mil_dim = backbone.hidden_dim

        if self.use_clinical:
            clin_emb_dim = max(32, hidden_dim // 2)
            self.clinical_mlp = nn.Sequential(
                nn.Linear(clinical_in_dim, clin_emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fusion = build_fusion(fusion_type, mil_dim, clin_emb_dim, hidden_dim, dropout)
            self.classifier = nn.Linear(hidden_dim, n_classes)
        else:
            self.classifier = nn.Linear(mil_dim, n_classes)
        self.apply(_init_weights)

    def forward(self, x, clinical=None):
        bag_repr = self.backbone(x)
        if self.use_clinical:
            if clinical is None:
                raise ValueError("use_clinical=True 时必须提供 clinical 特征")
            clin = clinical.unsqueeze(0) if clinical.dim() == 1 else clinical
            clin_emb = self.clinical_mlp(clin)
            fused = self.fusion(bag_repr, clin_emb)
            logits = self.classifier(fused)
        else:
            logits = self.classifier(bag_repr)
        return logits


class ClinicalOnlyModel(nn.Module):
    """仅临床信息的 MLP 二分类器（不使用 WSI 特征）。"""

    def __init__(self, clinical_in_dim, n_classes, hidden_dim=256, dropout=0.25):
        super().__init__()
        if clinical_in_dim <= 0:
            raise ValueError("ClinicalOnlyModel 需要 clinical_in_dim > 0")
        h = max(32, int(hidden_dim) // 2)
        self.net = nn.Sequential(
            nn.Linear(clinical_in_dim, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_classes),
        )
        self.apply(_init_weights)

    def forward(self, x=None, clinical=None):
        if clinical is None:
            raise ValueError("clinical_only 模式必须提供 clinical 特征")
        clin = clinical.unsqueeze(0) if clinical.dim() == 1 else clinical
        return self.net(clin)


def normalize_modality(cfg):
    """
    统一解析 modality / clinical_only / use_clinical。
    返回规范化后的 modality，并写回 cfg。
    """
    modality = cfg.get("modality", None)
    clinical_only = bool(cfg.get("clinical_only", False))
    use_clinical = cfg.get("use_clinical", True)

    if clinical_only:
        modality = "clinical"
    if modality is None or str(modality).strip() == "":
        modality = "pathomic" if use_clinical else "pathology"
    modality = str(modality).strip().lower()
    aliases = {
        "clin": "clinical",
        "clinical_only": "clinical",
        "path": "pathology",
        "wsi": "pathology",
        "fusion": "pathomic",
        "multi": "pathomic",
        "multimodal": "pathomic",
    }
    modality = aliases.get(modality, modality)
    if modality not in MODALITY_CHOICES:
        raise ValueError(f"未知 modality={modality!r}，可选: {list(MODALITY_CHOICES)}")

    cfg["modality"] = modality
    cfg["clinical_only"] = modality == "clinical"
    cfg["use_clinical"] = modality in ("pathomic", "clinical")
    return modality


def build_backbone(cfg):
    mt = cfg["model_type"]
    if mt == "abmil":
        return ABMILBackbone(cfg["in_dim"], dropout=cfg["drop_out"], hidden=cfg["hidden_dim"])
    if mt == "mean_mil":
        return MeanMaxBackbone(cfg["in_dim"], dropout=cfg["drop_out"],
                               hidden=cfg["hidden_dim"], pool="mean")
    if mt == "max_mil":
        return MeanMaxBackbone(cfg["in_dim"], dropout=cfg["drop_out"],
                               hidden=cfg["hidden_dim"], pool="max")
    if mt in ("mamba_mil", "trans_mil", "s4model"):
        # 作为特征 backbone：输出 bag 表征，供 pathomic 中期融合 / pathology 分类头使用
        return _build_repo_backbone(cfg)
    raise NotImplementedError(f"未知 model_type: {mt}")


def build_model(cfg, device):
    modality = normalize_modality(cfg)
    clinical_in_dim = int(cfg.get("clinical_in_dim", 0) or 0)

    if modality == "clinical":
        if clinical_in_dim <= 0:
            raise ValueError("modality=clinical 需要有效的 clinical_in_dim（请检查临床列）")
        # clinical_hidden_dim 优先，否则回退 hidden_dim
        h = int(cfg.get("clinical_hidden_dim") or cfg.get("hidden_dim") or 256)
        model = ClinicalOnlyModel(
            clinical_in_dim, cfg["n_classes"],
            hidden_dim=h, dropout=cfg["drop_out"],
        )
        return model.to(device)

    backbone = build_backbone(cfg)
    use_clinical = (modality == "pathomic") and clinical_in_dim > 0
    model = PathomicClassificationModel(
        backbone, cfg["n_classes"], clinical_in_dim,
        fusion_type=cfg.get("fusion_type", "concat"),
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["drop_out"],
        use_clinical=use_clinical,
    )
    return model.to(device)


class RepoMILBackbone(nn.Module):
    """
    将 MambaMIL / TransMIL / S4MIL 截到分类头之前，输出 bag 级表征 [1, 512]。
    可被 PathomicClassificationModel 接上临床中期融合。
    """

    REPO_HIDDEN = 512

    def __init__(self, inner, model_type):
        super().__init__()
        self.model = inner
        self.model_type = model_type
        self.hidden_dim = self.REPO_HIDDEN

    def forward(self, x):
        mt = self.model_type
        if mt == "mamba_mil":
            return self._encode_mamba(x)
        if mt == "trans_mil":
            return self._encode_trans(x)
        if mt == "s4model":
            return self._encode_s4(x)
        raise NotImplementedError(f"RepoMILBackbone 不支持 model_type={mt}")

    def _encode_mamba(self, x):
        m = self.model
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = m._fc1(x.float())
        if m.type == "SRMamba":
            for layer in m.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h, rate=m.rate)
                h = h + h_
        else:  # Mamba / BiMamba
            for layer in m.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h)
                h = h + h_
        h = m.norm(h)
        A = m.attention(h)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1)
        h = torch.bmm(A, h)  # [B, 1, 512]
        if h.size(0) == 1:
            h = h.squeeze(0)  # [1, 512]
        return h

    def _encode_trans(self, x):
        import numpy as _np

        m = self.model
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = m._fc1(x.float())
        H = h.shape[1]
        _H, _W = int(_np.ceil(_np.sqrt(H))), int(_np.ceil(_np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)
        cls_tokens = m.cls_token.expand(h.size(0), -1, -1).to(device=h.device, dtype=h.dtype)
        h = torch.cat((cls_tokens, h), dim=1)
        h = m.layer1(h)
        h = m.pos_layer(h, _H, _W)
        h = m.layer2(h)
        h = m.norm(h)[:, 0]  # [B, 512]
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return h

    def _encode_s4(self, x):
        m = self.model
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = m._fc1(x.float())
        h = m.s4_block(h)
        h = torch.max(h, dim=1).values  # [B, 512]
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return h


def _build_repo_backbone(cfg):
    """从同级 MambaMIL 仓库构建特征 backbone（输出 512-d bag 表征）。"""
    import sys

    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MambaMIL")
    repo_root = os.path.abspath(repo_root)
    if not os.path.isdir(repo_root):
        raise RuntimeError(
            f"未找到 MambaMIL 目录: {repo_root}\n"
            f"本仓库应自带 vendored 的 MambaMIL/；若缺失请恢复该目录或从 "
            f"https://github.com/isyangshu/MambaMIL 重新放入脚本同级。"
        )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    mt = cfg["model_type"]

    try:
        if mt == "mamba_mil":
            from models.MambaMIL import MambaMIL
            inner = MambaMIL(
                in_dim=cfg["in_dim"], n_classes=cfg["n_classes"],
                dropout=cfg["drop_out"], act="gelu", survival=False,
                layer=cfg["mambamil_layer"], rate=cfg["mambamil_rate"],
                type=cfg["mambamil_type"],
            )
        elif mt == "trans_mil":
            from models.TransMIL import TransMIL
            inner = TransMIL(
                cfg["in_dim"], cfg["n_classes"], dropout=cfg["drop_out"],
                act="relu", survival=False,
            )
        else:
            from models.S4MIL import S4Model
            inner = S4Model(
                in_dim=cfg["in_dim"], n_classes=cfg["n_classes"], act="gelu",
                dropout=cfg["drop_out"], survival=False,
            )
    except Exception as e:
        raise RuntimeError(
            f"无法加载模型 '{mt}'。请先安装 Mamba 依赖：\n"
            f"  bash scripts/install_mamba.sh\n"
            f"或改用 model_type=abmil。原始错误: {e}"
        )
    return RepoMILBackbone(inner, mt)


# ============================================================================
# 特征 IO / 临床编码 / 数据集
# ============================================================================
def _to_2d_float32(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"特征应为 2D [N_patch, dim]，得到 shape={arr.shape}")
    return arr


def load_features(path, feat_key="features"):
    """加载 .pt 或 .h5 特征，返回 float32 [N_patch, dim]。"""
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"特征文件不存在: {path}")

    if path.lower().endswith(".pt") or path.lower().endswith(".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return _to_2d_float32(obj.detach().cpu().numpy())
        if isinstance(obj, np.ndarray):
            return _to_2d_float32(obj)
        if isinstance(obj, dict):
            if feat_key in obj:
                val = obj[feat_key]
            elif "features" in obj:
                val = obj["features"]
            else:
                # 取第一个 tensor/ndarray
                val = next(iter(obj.values()))
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            return _to_2d_float32(val)
        raise TypeError(f"不支持的 .pt 内容类型: {type(obj)} ({path})")

    # 默认按 h5 处理
    if h5py is None:
        raise ImportError(f"读取 .h5 需要安装 h5py，当前文件: {path}")
    with h5py.File(path, "r") as f:
        key = feat_key if feat_key in f else list(f.keys())[0]
        feats = f[key][:]
    return _to_2d_float32(feats)


def detect_in_dim(feat_paths, feat_key):
    for p in feat_paths:
        if isinstance(p, str) and os.path.isfile(p):
            return int(load_features(p, feat_key).shape[-1])
    raise FileNotFoundError("未能找到任何可用的特征文件以推断特征维度。")


class ClinicalEncoder:
    """
    将 CSV 临床列编码为固定长度 float 向量：
      * 连续变量（Age/ER/PR/Ki67）：缺失填均值后做 z-score 标准化
      * 因子变量（Molecular/T/N/HER2）：按训练期类别表做 one-hot
    列类型由 KNOWN_NUMERIC / KNOWN_CATEGORICAL 显式指定，
    避免 T/N/HER2 因数值型 dtype 被误当作连续变量。
    """

    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_mean = {}
        self.numeric_std = {}
        self.cat_categories = {}
        self.output_dim = 0
        self.fitted = False

    def _detect_columns(self, df, clinical_cols):
        """按白名单显式区分因子变量与连续变量。"""
        numeric_cols, categorical_cols = [], []
        for col in clinical_cols:
            series = df[col]
            if col in KNOWN_CATEGORICAL:
                categorical_cols.append(col)
            elif col in KNOWN_NUMERIC:
                numeric_cols.append(col)
            elif series.dtype == object or str(series.dtype) == "category":
                categorical_cols.append(col)
            else:
                # 未声明类型时回退为连续变量
                numeric_cols.append(col)
        # 保持白名单顺序，便于复现与排查
        order = {c: i for i, c in enumerate(CLINICAL_WHITELIST)}
        self.numeric_cols = sorted(numeric_cols, key=lambda c: order.get(c, 999))
        self.categorical_cols = sorted(categorical_cols, key=lambda c: order.get(c, 999))

    @staticmethod
    def _normalize_cat_value(raw):
        if pd.isna(raw):
            return "missing"
        # 数值型因子（如 T/N/HER2）统一为去尾零的字符串，避免 "1.0" vs "1"
        if isinstance(raw, (int, np.integer)):
            return str(int(raw))
        if isinstance(raw, (float, np.floating)):
            if np.isnan(raw):
                return "missing"
            if float(raw).is_integer():
                return str(int(raw))
            return str(raw)
        s = str(raw).strip()
        return s if s else "missing"

    def fit(self, df):
        clinical_cols = get_clinical_columns(df)
        leaked = set(clinical_cols) & (META_COLS | EXCLUDED_CLINICAL)
        if leaked:
            raise ValueError(f"临床特征列包含禁止列: {sorted(leaked)}")
        if not clinical_cols:
            self.fitted = True
            self.output_dim = 0
            return self
        self._detect_columns(df, clinical_cols)
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce").astype(float)
            mean = float(vals.mean()) if vals.notna().any() else 0.0
            std = float(vals.std()) if vals.notna().any() else 1.0
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
            self.numeric_mean[col] = mean
            self.numeric_std[col] = std
        for col in self.categorical_cols:
            seen = {
                self._normalize_cat_value(v)
                for v in df[col].tolist()
            }
            preset = {
                self._normalize_cat_value(v)
                for v in KNOWN_CATEGORIES.get(col, [])
            }
            cats = sorted((seen | preset) - {"missing"})
            cats.append("missing")
            self.cat_categories[col] = cats
        self.output_dim = len(self.numeric_cols) + sum(
            len(self.cat_categories[c]) for c in self.categorical_cols
        )
        self.fitted = True
        return self

    def transform_row(self, row):
        if not self.fitted or self.output_dim == 0:
            return np.zeros((0,), dtype=np.float32)
        feats = []
        # 连续变量在前，因子 one-hot 在后（与 fit 时 output_dim 计算一致）
        for col in self.numeric_cols:
            val = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            val = float(val) if pd.notna(val) else self.numeric_mean[col]
            val = (val - self.numeric_mean[col]) / self.numeric_std[col]
            feats.append(val)
        for col in self.categorical_cols:
            val = self._normalize_cat_value(row.get(col, "missing"))
            cats = self.cat_categories[col]
            if val not in cats:
                val = "missing"
            onehot = [1.0 if val == c else 0.0 for c in cats]
            feats.extend(onehot)
        return np.asarray(feats, dtype=np.float32)

    def transform_df(self, pt_df):
        return np.stack([self.transform_row(pt_df.iloc[i]) for i in range(len(pt_df))], axis=0)

    def to_dict(self):
        return {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "cat_categories": self.cat_categories,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_dict(cls, d):
        enc = cls()
        enc.numeric_cols = list(d.get("numeric_cols", []))
        enc.categorical_cols = list(d.get("categorical_cols", []))
        enc.numeric_mean = {k: float(v) for k, v in d.get("numeric_mean", {}).items()}
        enc.numeric_std = {k: float(v) for k, v in d.get("numeric_std", {}).items()}
        enc.cat_categories = {k: list(v) for k, v in d.get("cat_categories", {}).items()}
        enc.output_dim = int(d.get("output_dim", 0))
        enc.fitted = True
        return enc


def save_clinical_encoder(encoder, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(encoder.to_dict(), f, ensure_ascii=False, indent=2)


def load_clinical_encoder(path):
    with open(path, "r", encoding="utf-8") as f:
        return ClinicalEncoder.from_dict(json.load(f))


def build_patient_table(df, label_col="label", feat_path_col=None, require_feats=True):
    """构建患者级表：每个 case 一行，含 y / feat_paths / 临床列。"""
    if label_col not in df.columns:
        raise KeyError(f"CSV 缺少标签列 {label_col}")

    df = df.copy()
    df["y"] = df[label_col].map(map_label)

    resolved_feat_col = None
    if require_feats:
        resolved_feat_col = resolve_feat_path_col(df, feat_path_col)
        df = df.dropna(subset=["case_id", resolved_feat_col, "y"])
    else:
        # 仅临床模式：特征路径可选
        if feat_path_col and feat_path_col in df.columns:
            resolved_feat_col = feat_path_col
        else:
            for col in ("slide_feats_path", "slide_feat_path"):
                if col in df.columns:
                    resolved_feat_col = col
                    break
        df = df.dropna(subset=["case_id", "y"])

    clinical_cols = get_clinical_columns(df)

    records = []
    for case_id, g in df.groupby("case_id", sort=False):
        labels = g["y"].astype(int).tolist()
        if len(set(labels)) != 1:
            raise ValueError(f"同一 case_id={case_id} 存在不一致 label: {labels}")
        if resolved_feat_col is not None:
            feat_paths = list(g[resolved_feat_col].astype(str))
        else:
            feat_paths = []
        rec = {
            "case_id": case_id,
            "y": int(labels[0]),
            "feat_paths": feat_paths,
        }
        for col in clinical_cols:
            rec[col] = g[col].iloc[0]
        records.append(rec)
    pt = pd.DataFrame(records).reset_index(drop=True)
    return pt, clinical_cols, resolved_feat_col


class PCRBagDataset(torch.utils.data.Dataset):
    """患者级数据集；每个样本返回 bag 特征 + 临床向量 + 二分类标签。"""

    def __init__(self, pt_df, feat_key, max_slides_train, training,
                 clinical_encoder=None, clinical_only=False):
        self.pt = pt_df.reset_index(drop=True)
        self.feat_key = feat_key
        self.max_slides_train = max_slides_train
        self.training = training
        self.clinical_encoder = clinical_encoder
        self.clinical_only = bool(clinical_only)
        if clinical_encoder is not None and clinical_encoder.output_dim > 0:
            self.clinical_matrix = clinical_encoder.transform_df(self.pt)
        else:
            self.clinical_matrix = None
        if self.clinical_only and self.clinical_matrix is None:
            raise ValueError("clinical_only 模式需要有效的 clinical_encoder")

    def __len__(self):
        return len(self.pt)

    def __getitem__(self, idx):
        row = self.pt.iloc[idx]
        if self.clinical_matrix is not None:
            clinical = torch.from_numpy(self.clinical_matrix[idx]).float()
        else:
            clinical = torch.zeros((0,), dtype=torch.float32)

        if self.clinical_only:
            # 占位特征，模型不会使用
            feats = torch.zeros((1, 1), dtype=torch.float32)
        else:
            paths = list(row["feat_paths"])
            if self.training and self.max_slides_train > 0 and len(paths) > self.max_slides_train:
                paths = random.sample(paths, self.max_slides_train)
            if not paths:
                raise ValueError(
                    f"case_id={row['case_id']} 无 slide 特征路径；"
                    f"病理/多模态模式需要 slide_feats_path"
                )
            arrs = [load_features(p, self.feat_key) for p in paths]
            feats = torch.from_numpy(np.concatenate(arrs, axis=0)).float()

        return feats, clinical, int(row["y"]), str(row["case_id"])


def collate_bag(batch):
    feats, clinical, label, cid = batch[0]
    return feats, clinical, label, cid


def make_loader(pt_df, cfg, training, clinical_encoder=None):
    clinical_only = bool(cfg.get("clinical_only", False)) or cfg.get("modality") == "clinical"
    ds = PCRBagDataset(
        pt_df, cfg["feat_key"], cfg["max_slides_train"], training,
        clinical_encoder=clinical_encoder, clinical_only=clinical_only,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=training, num_workers=cfg["num_workers"],
        collate_fn=collate_bag,
    )


def prepare_clinical_encoder(pt_train, cfg, out_dir=None):
    modality = normalize_modality(cfg)
    if modality == "pathology":
        cfg["clinical_in_dim"] = 0
        return None
    encoder = ClinicalEncoder().fit(pt_train)
    cfg["clinical_in_dim"] = int(encoder.output_dim)
    if cfg["clinical_in_dim"] <= 0:
        raise ValueError(
            f"modality={modality} 需要临床特征，但编码后维度为 0；"
            f"请检查 CSV 是否包含白名单列 {CLINICAL_WHITELIST}"
        )
    if out_dir is not None:
        save_clinical_encoder(encoder, os.path.join(out_dir, "clinical_encoder.json"))
    print(f"临床特征维度: {encoder.output_dim}  "
          f"(连续变量 {len(encoder.numeric_cols)}, 因子变量 {len(encoder.categorical_cols)})")
    if encoder.numeric_cols:
        print(f"  连续变量(标准化): {encoder.numeric_cols}")
    if encoder.categorical_cols:
        print(f"  因子变量(one-hot): {encoder.categorical_cols}")
        for col in encoder.categorical_cols:
            print(f"    {col}: {encoder.cat_categories.get(col, [])}")
    print(f"  白名单: {CLINICAL_WHITELIST}")
    print(f"  元信息/禁止列: {sorted(META_COLS | EXCLUDED_CLINICAL)}")
    return encoder


def model_forward(model, feats, clinical, cfg, device):
    modality = normalize_modality(cfg)
    if modality == "clinical":
        clinical = clinical.to(device, non_blocking=True)
        return model(None, clinical)
    if modality == "pathomic" and int(cfg.get("clinical_in_dim", 0) or 0) > 0:
        clinical = clinical.to(device, non_blocking=True)
        return model(feats, clinical)
    return model(feats, None)


# ============================================================================
# 指标
# ============================================================================
def _safe_div(num, den):
    return float(num / den) if den > 0 else float("nan")


def compute_cls_metrics(y_true, y_prob, y_pred=None, thr=0.5):
    """
    二分类指标（正类=pCR=1）。
    返回字段：
      n/n_pos/n_neg, tp/tn/fp/fn,
      auc, auprc, acc, balanced_acc,
      f1, precision, recall(=sensitivity), specificity,
      ppv(=precision), npv, mcc
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_pred is None:
        y_pred = (y_prob >= thr).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    n = int(len(y_true))
    metrics = {
        "n": n,
        "n_pos": int(np.sum(y_true == 1)) if n else 0,
        "n_neg": int(np.sum(y_true == 0)) if n else 0,
        "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        "auc": float("nan"),
        "auprc": float("nan"),
        "acc": float("nan"),
        "balanced_acc": float("nan"),
        "f1": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "ppv": float("nan"),
        "npv": float("nan"),
        "mcc": float("nan"),
    }
    if n == 0:
        return metrics

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) >= 2 else float("nan"),
    })
    # sensitivity = recall（正类召回）；specificity；PPV/NPV
    metrics["sensitivity"] = metrics["recall"]
    metrics["specificity"] = _safe_div(tn, tn + fp)
    metrics["ppv"] = metrics["precision"]
    metrics["npv"] = _safe_div(tn, tn + fn)

    # AUC / AUPRC 需要正负类都存在
    if len(np.unique(y_true)) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    return metrics


def _percentile_ci(values, ci=DEFAULT_BOOTSTRAP_CI):
    """对一组 bootstrap 统计量取百分位 CI；无有效值时返回 nan。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - float(ci)
    low = float(np.percentile(arr, 100.0 * alpha / 2.0))
    high = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def _attach_ci_fields(metrics, boot_store, n_boot, seed, level, ci=DEFAULT_BOOTSTRAP_CI):
    """将 bootstrap 百分位 CI 写入指标字典（{key}_ci95_low / {key}_ci95_high）。"""
    out = dict(metrics)
    for key in BOOTSTRAP_METRIC_KEYS:
        low, high = _percentile_ci(boot_store.get(key, []), ci=ci)
        out[f"{key}_ci95_low"] = low
        out[f"{key}_ci95_high"] = high
    out["bootstrap_n"] = int(n_boot)
    out["bootstrap_seed"] = int(seed)
    out["bootstrap_level"] = str(level)
    out["bootstrap_ci"] = float(ci)
    return out


def compute_cls_metrics_bootstrap(
    y_true, y_prob, y_pred=None, thr=0.5,
    n_boot=DEFAULT_N_BOOT, seed=1, ci=DEFAULT_BOOTSTRAP_CI,
):
    """
    点估计 + 样本级 bootstrap 95% CI。
    对 n 个样本有放回重采样 n_boot 次，对 BOOTSTRAP_METRIC_KEYS 取百分位区间。
    n_boot<=0 或样本不足时仅返回点估计。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_pred is None:
        y_pred = (y_prob >= float(thr)).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    point = compute_cls_metrics(y_true, y_prob, y_pred, thr=thr)
    n_boot = int(n_boot)
    if n_boot <= 0 or point["n"] < 1:
        return point

    n = int(point["n"])
    rng = np.random.RandomState(int(seed))
    boot_store = {k: [] for k in BOOTSTRAP_METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        m = compute_cls_metrics(y_true[idx], y_prob[idx], y_pred[idx], thr=thr)
        for k in BOOTSTRAP_METRIC_KEYS:
            v = m.get(k, float("nan"))
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v == v:  # not NaN
                boot_store[k].append(v)
    return _attach_ci_fields(point, boot_store, n_boot, seed, level="sample", ci=ci)


def _mean_ignore_nan(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def aggregate_fold_metrics_bootstrap(
    fold_bundles,
    n_boot=DEFAULT_N_BOOT,
    seed=1,
    ci=DEFAULT_BOOTSTRAP_CI,
    molecular_cats=None,
):
    """
    对 K 折指标做算术平均，并对折索引做 bootstrap×n_boot 给出均值的 95% CI。
    fold_bundles: list[{"overall": dict, "by_molecular": dict}, ...]
    """
    bundles = [b for b in (fold_bundles or []) if isinstance(b, dict) and b.get("overall")]
    empty = compute_cls_metrics([], [], [])
    cats = list(molecular_cats or MOLECULAR_CATEGORIES)
    if not bundles:
        return {
            "overall": empty,
            "by_molecular": {c: dict(empty) for c in cats},
            "n_folds": 0,
            "bootstrap_level": "fold",
        }

    # 收集 overall 各折点估计
    overall_list = [b["overall"] for b in bundles]
    n_folds = len(overall_list)

    def _aggregate_metric_dicts(dicts, level_seed):
        """对一组同构指标字典求均值 + 折级 bootstrap CI。"""
        if not dicts:
            return dict(empty)
        # 点估计：各折算术平均（计数类也平均，便于报告）
        keys = set()
        for d in dicts:
            keys.update(d.keys())
        # 排除已有 CI / bootstrap 元数据，避免把 CI 再平均
        skip_suffix = ("_ci95_low", "_ci95_high")
        skip_exact = {
            "bootstrap_n", "bootstrap_seed", "bootstrap_level", "bootstrap_ci",
        }
        mean_metrics = {}
        for k in sorted(keys):
            if k in skip_exact or k.endswith(skip_suffix):
                continue
            vals = []
            for d in dicts:
                if k not in d:
                    continue
                try:
                    vals.append(float(d[k]))
                except (TypeError, ValueError):
                    continue
            mean_metrics[k] = _mean_ignore_nan(vals)

        n_boot_i = int(n_boot)
        if n_boot_i <= 0 or len(dicts) < 1:
            mean_metrics["n_folds"] = int(len(dicts))
            return mean_metrics

        rng = np.random.RandomState(int(level_seed))
        boot_store = {k: [] for k in BOOTSTRAP_METRIC_KEYS}
        n = len(dicts)
        for _ in range(n_boot_i):
            idx = rng.randint(0, n, size=n)
            for k in BOOTSTRAP_METRIC_KEYS:
                vals = []
                for i in idx:
                    v = dicts[i].get(k, float("nan"))
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                    if v == v:
                        vals.append(v)
                if vals:
                    boot_store[k].append(float(np.mean(vals)))
        out = _attach_ci_fields(
            mean_metrics, boot_store, n_boot_i, level_seed, level="fold", ci=ci
        )
        out["n_folds"] = int(n)
        return out

    overall = _aggregate_metric_dicts(overall_list, seed)

    # 亚组：仅对 n>0 的折参与平均与 bootstrap
    all_cats = list(cats)
    for b in bundles:
        for c in (b.get("by_molecular") or {}):
            if c not in all_cats:
                all_cats.append(c)

    by_mol = {}
    for j, cat in enumerate(all_cats):
        cat_dicts = []
        for b in bundles:
            m = (b.get("by_molecular") or {}).get(cat)
            if m is None:
                continue
            if int(m.get("n", 0) or 0) <= 0:
                continue
            cat_dicts.append(m)
        by_mol[cat] = _aggregate_metric_dicts(cat_dicts, int(seed) + 1 + j)

    return {
        "overall": overall,
        "by_molecular": by_mol,
        "n_folds": n_folds,
        "bootstrap_level": "fold",
        "bootstrap_n": int(n_boot),
        "bootstrap_seed": int(seed),
        "bootstrap_ci": float(ci),
    }


def prefix_epoch_metrics(metrics, prefix):
    """将 run_epoch 指标扁平化为 train_*/val_* 列，用于 metrics.csv。"""
    out = {}
    for k in EPOCH_METRIC_KEYS:
        if k in metrics:
            out[f"{prefix}{k}"] = metrics[k]
    return out


def normalize_molecular_value(raw):
    """将 Molecular 原始值规范化到预定义四类（无法识别则保留去空白字符串）。"""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or pd.isna(raw):
        return "missing"
    s = str(raw).strip()
    if not s:
        return "missing"
    aliases = {
        "hr+her2-": "HR+HER2-",
        "hr+her2+": "HR+HER2+",
        "tnbc": "TNBC",
        "her2": "HER2",
        "her2+": "HER2",
        "her2 enriched": "HER2",
        "her2-positive": "HER2",
        "triple negative": "TNBC",
        "luminal": "HR+HER2-",
        "luminal a": "HR+HER2-",
        "luminal b": "HR+HER2-",
        "luminal her2+": "HR+HER2+",
        "hr+": "HR+HER2-",
    }
    mapped = aliases.get(s.lower())
    if mapped is not None:
        return mapped
    for cat in MOLECULAR_CATEGORIES:
        if s == cat:
            return cat
    return s


def resolve_bootstrap_params(cfg=None, n_boot=None, bootstrap_seed=None, ci=None):
    """从 cfg / 显式参数解析 bootstrap 设置。"""
    cfg = cfg or {}
    if n_boot is None:
        n_boot = cfg.get("n_boot", DEFAULT_N_BOOT)
    if bootstrap_seed is None:
        bootstrap_seed = cfg.get("seed", 1)
    if ci is None:
        ci = cfg.get("bootstrap_ci", DEFAULT_BOOTSTRAP_CI)
    return int(n_boot), int(bootstrap_seed), float(ci)


def compute_metrics_with_subgroups(
    pred_df, thr=0.5, molecular_cats=None,
    n_boot=DEFAULT_N_BOOT, bootstrap_seed=1, ci=DEFAULT_BOOTSTRAP_CI,
):
    """
    基于预测表计算总体指标 + 各 Molecular 亚组指标（含样本级 bootstrap 95% CI）。
    pred_df 需含列: label, prob_pCR；可选 Molecular / pred。
    """
    metric_fn = (
        (lambda yt, yp, yhat, t: compute_cls_metrics_bootstrap(
            yt, yp, yhat, thr=t, n_boot=n_boot, seed=bootstrap_seed, ci=ci
        ))
        if int(n_boot) > 0 else
        (lambda yt, yp, yhat, t: compute_cls_metrics(yt, yp, yhat, thr=t))
    )

    if pred_df is None or len(pred_df) == 0:
        empty = metric_fn([], [], None, thr)
        cats = list(molecular_cats or MOLECULAR_CATEGORIES)
        return {
            "overall": empty,
            "by_molecular": {c: metric_fn([], [], None, thr) for c in cats},
        }

    df = pred_df.copy()
    if "pred" not in df.columns:
        df["pred"] = (df["prob_pCR"].astype(float) >= thr).astype(int)
    if "Molecular" in df.columns:
        df["Molecular"] = df["Molecular"].map(normalize_molecular_value)
    else:
        df["Molecular"] = "missing"

    overall = metric_fn(df["label"], df["prob_pCR"], df["pred"], thr)
    cats = list(molecular_cats or MOLECULAR_CATEGORIES)
    # 保证四类都有条目；额外出现的类别也一并报告
    extra = [c for c in sorted(df["Molecular"].unique()) if c not in cats]
    by_mol = {}
    for j, cat in enumerate(cats + extra):
        sub = df[df["Molecular"] == cat]
        # 亚组使用偏移 seed，避免与 overall 完全共用同一重采样序列
        if int(n_boot) > 0:
            by_mol[cat] = compute_cls_metrics_bootstrap(
                sub["label"], sub["prob_pCR"],
                sub["pred"] if len(sub) else None,
                thr=thr, n_boot=n_boot,
                seed=int(bootstrap_seed) + 1 + j, ci=ci,
            )
        else:
            by_mol[cat] = compute_cls_metrics(
                sub["label"], sub["prob_pCR"],
                sub["pred"] if len(sub) else None, thr=thr,
            )
    return {"overall": overall, "by_molecular": by_mol}


def save_prediction_outputs(
    out_dir, pred_df, prefix="val", extra_meta=None,
    n_boot=None, bootstrap_seed=None, ci=None, cfg=None,
):
    """保存逐样本预测 CSV + 总体/亚组指标 JSON/YAML（含样本级 bootstrap CI）。"""
    n_boot, bootstrap_seed, ci = resolve_bootstrap_params(
        cfg=cfg, n_boot=n_boot, bootstrap_seed=bootstrap_seed, ci=ci
    )
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, f"{prefix}_predictions.csv")
    metrics_json = os.path.join(out_dir, f"{prefix}_metrics.json")
    metrics_yaml = os.path.join(out_dir, f"{prefix}_metrics.yaml")

    if pred_df is None or len(pred_df) == 0:
        empty_df = pd.DataFrame(
            columns=["case_id", "label", "Molecular", "prob_pCR", "pred"]
        )
        empty_df.to_csv(pred_path, index=False)
        metrics = compute_metrics_with_subgroups(
            empty_df, n_boot=n_boot, bootstrap_seed=bootstrap_seed, ci=ci
        )
    else:
        out_df = pred_df.copy()
        if "Molecular" in out_df.columns:
            out_df["Molecular"] = out_df["Molecular"].map(normalize_molecular_value)
        cols = [c for c in ["case_id", "label", "Molecular", "prob_pCR", "pred", "fold"]
                if c in out_df.columns]
        out_df[cols].to_csv(pred_path, index=False)
        metrics = compute_metrics_with_subgroups(
            out_df, n_boot=n_boot, bootstrap_seed=bootstrap_seed, ci=ci
        )

    if extra_meta:
        metrics = dict(metrics)
        metrics.update(extra_meta)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(metrics), f, ensure_ascii=False, indent=2)
    save_yaml(metrics, metrics_yaml)
    return pred_path, metrics


def _fmt_metric(m, key, nd=4):
    v = m.get(key, float("nan"))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "nan"
    if v != v:
        return "nan"
    return f"{v:.{nd}f}"


def _fmt_metric_ci(m, key, nd=4):
    """格式化为 value [low, high]；无 CI 时仅 value。"""
    base = _fmt_metric(m, key, nd=nd)
    low = m.get(f"{key}_ci95_low", float("nan"))
    high = m.get(f"{key}_ci95_high", float("nan"))
    try:
        low_f, high_f = float(low), float(high)
    except (TypeError, ValueError):
        return base
    if low_f != low_f or high_f != high_f:
        return base
    return f"{base} [{low_f:.{nd}f}, {high_f:.{nd}f}]"


def print_metrics_block(title, metrics_bundle):
    """打印总体 + 亚组指标摘要（含 95% CI，若有）。"""
    overall = metrics_bundle.get("overall", {})
    print(
        f"{title} overall: n={overall.get('n', 0)}, "
        f"AUC={_fmt_metric_ci(overall, 'auc')}, AUPRC={_fmt_metric_ci(overall, 'auprc')}, "
        f"ACC={_fmt_metric_ci(overall, 'acc')}, bACC={_fmt_metric_ci(overall, 'balanced_acc')}, "
        f"F1={_fmt_metric_ci(overall, 'f1')}, "
        f"P={_fmt_metric_ci(overall, 'precision')}, R={_fmt_metric_ci(overall, 'recall')}, "
        f"Spec={_fmt_metric_ci(overall, 'specificity')}, MCC={_fmt_metric_ci(overall, 'mcc')}"
    )
    print(
        f"  confusion: TP={overall.get('tp', 0)} TN={overall.get('tn', 0)} "
        f"FP={overall.get('fp', 0)} FN={overall.get('fn', 0)} "
        f"PPV={_fmt_metric_ci(overall, 'ppv')} NPV={_fmt_metric_ci(overall, 'npv')}"
    )
    by_mol = metrics_bundle.get("by_molecular", {})
    for cat in MOLECULAR_CATEGORIES:
        m = by_mol.get(cat)
        if not m or int(m.get("n", 0) or 0) == 0:
            continue
        print(
            f"  [{cat}] n={m['n']} AUC={_fmt_metric_ci(m, 'auc')} "
            f"AUPRC={_fmt_metric_ci(m, 'auprc')} ACC={_fmt_metric_ci(m, 'acc')} "
            f"F1={_fmt_metric_ci(m, 'f1')} P={_fmt_metric_ci(m, 'precision')} "
            f"R={_fmt_metric_ci(m, 'recall')} Spec={_fmt_metric_ci(m, 'specificity')} "
            f"MCC={_fmt_metric_ci(m, 'mcc')}"
        )


def resolve_overall_splits_path(splits_path, explicit=None):
    """
    解析亚组专训对应的总体划分路径。
    优先 explicit；否则若当前划分位于 molecular_subgroups/<name>/ 下，
    自动上溯到父目录的 kfold_splits.yaml。
    """
    if explicit:
        try:
            return resolve_splits_path(explicit, required=True)
        except Exception as e:
            print(f"警告: 无法解析 --eval_overall_splits={explicit}: {e}")
            return None
    if not splits_path:
        return None
    path = os.path.abspath(str(splits_path))
    base = path if os.path.isdir(path) else os.path.dirname(path)
    # .../molecular_subgroups/<Molecular>/[kfold_splits.yaml]
    parts = base.rstrip(os.sep).split(os.sep)
    if "molecular_subgroups" in parts:
        idx = parts.index("molecular_subgroups")
        parent_dir = os.sep.join(parts[:idx])
        for name in ("kfold_splits.yaml", "kfold_splits.yml", "kfold_splits.json"):
            cand = os.path.join(parent_dir, name)
            if os.path.isfile(cand):
                return cand
    return None


def is_subgroup_split_meta(split_meta):
    if not isinstance(split_meta, dict):
        return False
    if str(split_meta.get("split_type", "")).lower() == "subgroup":
        return True
    if split_meta.get("subgroup"):
        return True
    return False


# ============================================================================
# 训练 / 验证
# ============================================================================
def run_epoch(model, loader, optimizer, cfg, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    ys, probs, preds = [], [], []
    gc = max(1, int(cfg["gc"]))
    criterion = nn.CrossEntropyLoss()
    if train:
        optimizer.zero_grad()

    for i, (feats, clinical, label, _cid) in enumerate(loader):
        feats = feats.to(device, non_blocking=True)
        label_t = torch.tensor([label], device=device, dtype=torch.long)

        with torch.set_grad_enabled(train):
            logits = model_forward(model, feats, clinical, cfg, device)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            loss = criterion(logits, label_t)

        if train:
            (loss / gc).backward()
            if (i + 1) % gc == 0 or (i + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        prob1 = float(torch.softmax(logits, dim=-1)[0, 1].detach().cpu().item())
        pred = int(torch.argmax(logits, dim=-1).detach().cpu().item())
        ys.append(int(label))
        probs.append(prob1)
        preds.append(pred)

    avg_loss = total_loss / max(1, len(loader))
    metrics = compute_cls_metrics(ys, probs, preds)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def predict_patient_table(model, pt_df, cfg, device, clinical_encoder=None):
    """对患者表逐例评估预测，返回含 case_id/label/Molecular/prob_pCR/pred 的 DataFrame。"""
    if pt_df is None or len(pt_df) == 0:
        return pd.DataFrame(
            columns=["case_id", "label", "Molecular", "prob_pCR", "pred"]
        )
    model.eval()
    modality = normalize_modality(cfg)
    loader = make_loader(pt_df, cfg, training=False, clinical_encoder=clinical_encoder)
    mol_map = {}
    if "Molecular" in pt_df.columns:
        for _, row in pt_df.iterrows():
            mol_map[str(row["case_id"])] = normalize_molecular_value(row["Molecular"])

    rows = []
    for feats, clinical, label, cid in loader:
        if modality != "clinical":
            feats = feats.to(device, non_blocking=True)
        logits = model_forward(model, feats, clinical, cfg, device)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        prob1 = float(torch.softmax(logits, dim=-1)[0, 1].detach().cpu().item())
        pred = int(torch.argmax(logits, dim=-1).detach().cpu().item())
        rows.append({
            "case_id": str(cid),
            "label": int(label),
            "Molecular": mol_map.get(str(cid), "missing"),
            "prob_pCR": prob1,
            "pred": pred,
        })
    return pd.DataFrame(rows)


def load_model_for_eval(cfg, device, ckpt_path, encoder=None):
    """构建模型并加载权重；按需根据 encoder 设置 clinical_in_dim。"""
    cfg = dict(cfg)
    modality = normalize_modality(cfg)
    if encoder is not None:
        cfg["clinical_in_dim"] = int(encoder.output_dim)
    elif modality in ("pathomic", "clinical"):
        cfg.setdefault("clinical_in_dim", int(cfg.get("clinical_in_dim", 0) or 0))
    model = build_model(cfg, device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def resolve_ckpt_path(fold_dir, prefer_best=True, preferred_name=None):
    """在 fold 目录中寻找可用权重。"""
    candidates = []
    if preferred_name:
        candidates.append(preferred_name)
    if prefer_best:
        candidates.extend([
            "checkpoint_best.pt",
            "checkpoint_best_loss.pt",
            "checkpoint_last.pt",
        ])
    else:
        candidates.extend([
            "checkpoint_last.pt",
            "checkpoint_best.pt",
            "checkpoint_best_loss.pt",
        ])
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        path = os.path.join(fold_dir, name)
        if os.path.isfile(path):
            return path
    return None


def train_one_run(pt_train, pt_val, cfg, device, out_dir, fold_tag=""):
    """训练单个 run（一个 fold 或 all_train）。返回该 run 的历史与最佳信息。"""
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["seed"])

    encoder = prepare_clinical_encoder(pt_train, cfg, out_dir)
    model = build_model(cfg, device)
    optimizer = get_optimizer(model, cfg)
    train_loader = make_loader(pt_train, cfg, training=True, clinical_encoder=encoder)
    val_loader = (
        make_loader(pt_val, cfg, training=False, clinical_encoder=encoder)
        if pt_val is not None else None
    )

    history = []
    best_auc, best_epoch = -1.0, -1
    best_loss = float("inf")
    ckpt_best = os.path.join(out_dir, "checkpoint_best.pt")
    ckpt_best_loss = os.path.join(out_dir, "checkpoint_best_loss.pt")
    ckpt_last = os.path.join(out_dir, "checkpoint_last.pt")

    # 早停：仅在有验证集时生效
    early_stop = bool(cfg.get("early_stop", True)) and (val_loader is not None)
    patience = max(1, int(cfg.get("patience", 10)))
    min_delta = float(cfg.get("min_delta", 0.0))
    stop_metric = str(cfg.get("early_stop_metric", "val_auc") or "val_auc").strip().lower()
    if stop_metric not in EARLY_STOP_METRIC_CHOICES:
        print(f"警告: 未知 early_stop_metric={stop_metric!r}，回退为 val_auc")
        stop_metric = "val_auc"
    # loss 越小越好；其余指标越大越好
    maximize = stop_metric != "val_loss"
    best_monitor = -float("inf") if maximize else float("inf")
    bad_epochs = 0
    early_stopped = False
    stopped_epoch = None

    if early_stop:
        print(
            f"[{fold_tag}] 早停已启用: metric={stop_metric}, "
            f"patience={patience}, min_delta={min_delta}"
        )
    elif val_loader is None and bool(cfg.get("early_stop", True)):
        print(f"[{fold_tag}] 无验证集，跳过早停（all_train）")

    for epoch in range(cfg["max_epochs"]):
        tr = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        rec = {"epoch": epoch}
        rec.update(prefix_epoch_metrics(tr, "train_"))
        if val_loader is not None:
            va = run_epoch(model, val_loader, optimizer, cfg, device, train=False)
            rec.update(prefix_epoch_metrics(va, "val_"))
            # 选模仍优先按 val AUC；AUC 不可用时回退到 val ACC
            score = va["auc"] if va["auc"] == va["auc"] else va["acc"]
            if score == score and score > best_auc:
                best_auc, best_epoch = float(score), epoch
                torch.save(model.state_dict(), ckpt_best)

            # 早停监控（va 字典键无 val_ 前缀）
            if early_stop:
                metric_key = (
                    stop_metric[4:] if stop_metric.startswith("val_") else stop_metric
                )
                monitor = va.get(metric_key, float("nan"))
                improved = False
                if monitor == monitor:  # not NaN
                    if maximize:
                        improved = monitor > (best_monitor + min_delta)
                    else:
                        improved = monitor < (best_monitor - min_delta)
                if improved:
                    best_monitor = float(monitor)
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                rec["early_stop_monitor"] = (
                    float(monitor) if monitor == monitor else None
                )
                rec["early_stop_bad_epochs"] = bad_epochs

            print(
                f"[{fold_tag} epoch {epoch}] "
                f"train_loss={tr['loss']:.4f} train_auc={tr['auc']:.4f} "
                f"val_loss={va['loss']:.4f} val_auc={va['auc']:.4f} "
                f"val_acc={va['acc']:.4f} val_f1={va['f1']:.4f} "
                f"val_P={va['precision']:.4f} val_R={va['recall']:.4f} "
                f"val_spec={va['specificity']:.4f}"
                + (f" bad={bad_epochs}/{patience}" if early_stop else "")
            )
        else:
            if tr["loss"] < best_loss:
                best_loss, best_epoch = tr["loss"], epoch
                torch.save(model.state_dict(), ckpt_best_loss)
            print(
                f"[{fold_tag} epoch {epoch}] "
                f"train_loss={tr['loss']:.4f} train_auc={tr['auc']:.4f} "
                f"train_acc={tr['acc']:.4f} train_f1={tr['f1']:.4f} "
                f"train_P={tr['precision']:.4f} train_R={tr['recall']:.4f}"
            )

        history.append(rec)

        if early_stop and bad_epochs >= patience:
            early_stopped = True
            stopped_epoch = epoch
            print(
                f"[{fold_tag}] 早停触发于 epoch {epoch} "
                f"(metric={stop_metric}, best={best_monitor:.4f}, "
                f"patience={patience}, best_ckpt_epoch={best_epoch})"
            )
            break

    torch.save(model.state_dict(), ckpt_last)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    early_stop_info = {
        "enabled": bool(early_stop),
        "metric": stop_metric if early_stop else None,
        "patience": patience if early_stop else None,
        "min_delta": min_delta if early_stop else None,
        "best_monitor": (
            float(best_monitor)
            if early_stop and best_monitor not in (float("inf"), -float("inf"))
            else None
        ),
        "early_stopped": bool(early_stopped),
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
    }
    with open(os.path.join(out_dir, "early_stop.json"), "w", encoding="utf-8") as f:
        json.dump(_to_builtin(early_stop_info), f, ensure_ascii=False, indent=2)

    # 用最佳权重对验证集做逐样本评估并落盘
    val_pred_df = None
    val_metrics = None
    if pt_val is not None and len(pt_val) > 0:
        ckpt_for_eval = ckpt_best if os.path.isfile(ckpt_best) else ckpt_last
        eval_model, eval_cfg = load_model_for_eval(
            cfg, device, ckpt_for_eval, encoder=encoder
        )
        val_pred_df = predict_patient_table(
            eval_model, pt_val, eval_cfg, device, clinical_encoder=encoder
        )
        _, val_metrics = save_prediction_outputs(
            out_dir, val_pred_df, prefix="val",
            extra_meta={
                "ckpt_path": os.path.abspath(ckpt_for_eval),
                "best_epoch": best_epoch,
                "split": "fold_val",
                "early_stop": early_stop_info,
            },
            cfg=cfg,
        )
        print_metrics_block(f"[{fold_tag}] val", val_metrics)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_auc": best_auc if val_loader is not None else None,
        "best_loss": best_loss if val_loader is None else None,
        "early_stop": early_stop_info,
        "encoder": encoder,
        "val_predictions": val_pred_df,
        "val_metrics": val_metrics,
        "ckpt_best": ckpt_best if os.path.isfile(ckpt_best) else None,
        "ckpt_last": ckpt_last,
    }


def _molecular_series(pt):
    """返回患者级 Molecular 字符串序列；缺失记为 missing。"""
    if "Molecular" not in pt.columns:
        return None
    return pt["Molecular"].fillna("missing").astype(str).str.strip().replace("", "missing")


def _value_counts_dict(series):
    if series is None:
        return {}
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def normalize_stratify_by(value):
    """规范化 stratify_by 超参取值。"""
    if value is None:
        return "Molecular_label"
    key = str(value).strip()
    aliases = {
        "molecular_label": "Molecular_label",
        "molecular+label": "Molecular_label",
        "label+molecular": "Molecular_label",
        "label_molecular": "Molecular_label",
        "mol_label": "Molecular_label",
        "joint": "Molecular_label",
        "molecular": "Molecular",
        "mol": "Molecular",
        "y": "label",
        "random": "none",
        "kfold_random": "none",
        "nonstratified": "none",
    }
    key_norm = aliases.get(key.lower(), key)
    if key_norm not in STRATIFY_BY_CHOICES:
        raise ValueError(
            f"未知 stratify_by={value!r}，可选: {list(STRATIFY_BY_CHOICES)}"
        )
    return key_norm


def build_kfold_stratify_labels(pt, stratify_by="Molecular_label"):
    """
    按超参 stratify_by 构造分层标签。
    返回 (stratify_array_or_None, requested_strategy)。
    stratify_array 为 None 表示不分层（普通 KFold）。
    """
    strategy = normalize_stratify_by(stratify_by)
    y = pt["y"].astype(int)
    mol = _molecular_series(pt)

    if strategy == "none":
        print("K 折不分层（stratify_by=none），使用普通随机 KFold")
        return None, strategy

    if strategy == "label":
        print(f"K 折按 label 分层，分布: {_value_counts_dict(y.astype(str))}")
        return y.values, strategy

    if strategy == "Molecular":
        if mol is None:
            raise ValueError("stratify_by=Molecular 但患者表缺少 Molecular 列")
        print(f"K 折按 Molecular 分层，分布: {_value_counts_dict(mol)}")
        return mol.values, strategy

    # Molecular_label：label 与 Molecular 联合分层
    if mol is None:
        raise ValueError("stratify_by=Molecular_label 但患者表缺少 Molecular 列")
    joint = mol.astype(str) + "|y" + y.astype(str)
    print(
        f"K 折按 Molecular+label 联合分层，联合分布: {_value_counts_dict(joint)}"
    )
    return joint.values, strategy


def make_kfold_splits(pt, cfg):
    """
    生成 K 折索引。
    分层依据由 cfg['stratify_by'] 控制：
      Molecular_label / Molecular / label / none
    若请求的分层因样本量不足失败，则按
      Molecular_label -> Molecular -> label -> none
    依次回退，并在 meta 中记录 requested / actual。
    返回 (splits, split_meta)。
    """
    n_splits = int(cfg["k"])
    seed = int(cfg["seed"])
    case_ids = pt["case_id"].values
    requested = normalize_stratify_by(cfg.get("stratify_by", "Molecular_label"))
    cfg["stratify_by"] = requested  # 规范化后写回，确保 config.yaml 一致

    def _try_stratified(labels, name):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(case_ids, labels))
        return splits, name

    def _random_split():
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(case_ids)), "none"

    # 回退链：从请求策略开始，去掉已更“强”的前置项
    fallback_order = ["Molecular_label", "Molecular", "label", "none"]
    start = fallback_order.index(requested)
    chain = fallback_order[start:]

    splits, actual = None, None
    errors = []
    for cand in chain:
        try:
            if cand == "none":
                splits, actual = _random_split()
            else:
                labels, name = build_kfold_stratify_labels(pt, cand)
                splits, actual = _try_stratified(labels, name)
            if cand != requested:
                print(f"警告: stratify_by={requested} 不可用，已回退为 {actual}")
            break
        except ValueError as e:
            errors.append(f"{cand}: {e}")
            print(f"警告: 按 {cand} 划分失败（{e}）")
            continue

    if splits is None:
        raise RuntimeError(
            "无法完成 K 折划分，尝试记录: " + " | ".join(errors)
        )

    mol = _molecular_series(pt)
    joint = None
    if mol is not None:
        joint = mol.astype(str) + "|y" + pt["y"].astype(int).astype(str)

    meta = {
        "split_mode": "kfold",
        "k": n_splits,
        "seed": seed,
        "stratify_by": actual,                 # 实际使用的划分依据
        "stratify_by_requested": requested,    # 超参请求的划分依据
        "stratify_fallback": actual != requested,
        "n_cases": int(len(pt)),
        "overall_molecular": _value_counts_dict(mol),
        "overall_label": {
            "N-pCR(0)": int((pt["y"] == 0).sum()),
            "pCR(1)": int((pt["y"] == 1).sum()),
        },
        "overall_molecular_label": _value_counts_dict(joint),
    }
    print(
        f"K 折划分策略: stratify_by={actual}"
        f"{'' if actual == requested else f' (requested={requested})'}, "
        f"k={n_splits}, seed={seed}"
    )
    return splits, meta


def build_fold_split_record(pt, fold, tr_idx, va_idx, stratify_by):
    """构造单折划分记录（含 case_id / Molecular / label 分布）。"""
    pt_tr = pt.iloc[tr_idx]
    pt_va = pt.iloc[va_idx]
    mol_all = _molecular_series(pt)
    mol_tr = mol_all.iloc[tr_idx] if mol_all is not None else None
    mol_va = mol_all.iloc[va_idx] if mol_all is not None else None

    train_cases = []
    for i in tr_idx:
        row = pt.iloc[int(i)]
        train_cases.append({
            "case_id": str(row["case_id"]),
            "label": int(row["y"]),
            "Molecular": (
                str(mol_all.iloc[int(i)]) if mol_all is not None else None
            ),
        })
    val_cases = []
    for i in va_idx:
        row = pt.iloc[int(i)]
        val_cases.append({
            "case_id": str(row["case_id"]),
            "label": int(row["y"]),
            "Molecular": (
                str(mol_all.iloc[int(i)]) if mol_all is not None else None
            ),
        })

    return {
        "fold": int(fold),
        "stratify_by": stratify_by,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "train_case_ids": [c["case_id"] for c in train_cases],
        "val_case_ids": [c["case_id"] for c in val_cases],
        "train_cases": train_cases,
        "val_cases": val_cases,
        "train_molecular": _value_counts_dict(mol_tr),
        "val_molecular": _value_counts_dict(mol_va),
        "train_label": {
            "N-pCR(0)": int((pt_tr["y"] == 0).sum()),
            "pCR(1)": int((pt_tr["y"] == 1).sum()),
        },
        "val_label": {
            "N-pCR(0)": int((pt_va["y"] == 0).sum()),
            "pCR(1)": int((pt_va["y"] == 1).sum()),
        },
        "train_molecular_label": _value_counts_dict(
            (mol_tr.astype(str) + "|y" + pt_tr["y"].astype(int).astype(str))
            if mol_tr is not None else None
        ),
        "val_molecular_label": _value_counts_dict(
            (mol_va.astype(str) + "|y" + pt_va["y"].astype(int).astype(str))
            if mol_va is not None else None
        ),
    }


def save_kfold_splits(log_dir, split_meta, fold_records):
    """将完整 K 折划分写入日志：总表 + 各 fold 子目录。"""
    os.makedirs(log_dir, exist_ok=True)
    payload = dict(split_meta)
    payload["folds"] = fold_records
    splits_path = os.path.join(log_dir, "kfold_splits.yaml")
    save_yaml(payload, splits_path)
    # 同步一份 json，便于程序读取
    with open(os.path.join(log_dir, "kfold_splits.json"), "w", encoding="utf-8") as f:
        json.dump(_to_builtin(payload), f, ensure_ascii=False, indent=2)

    for rec in fold_records:
        fold_dir = os.path.join(log_dir, f"fold_{rec['fold']}")
        os.makedirs(fold_dir, exist_ok=True)
        save_yaml(rec, os.path.join(fold_dir, "split.yaml"))
        # 便于快速查看的 case_id 列表
        with open(os.path.join(fold_dir, "train_case_ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rec["train_case_ids"]) + ("\n" if rec["train_case_ids"] else ""))
        with open(os.path.join(fold_dir, "val_case_ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rec["val_case_ids"]) + ("\n" if rec["val_case_ids"] else ""))

    print(f"K 折划分已保存: {splits_path}")
    return splits_path


def resolve_splits_path(splits_path, required=True):
    """将目录或文件路径解析为 kfold_splits.yaml/.json。"""
    if splits_path is None or str(splits_path).strip() == "":
        if required:
            raise ValueError(
                "必须提供 --splits_path（预划分结果）。"
                "请先运行: python make_kfold_splits.py --csv_path ... --out_dir ..."
            )
        return None
    path = os.path.abspath(str(splits_path))
    if os.path.isdir(path):
        for name in ("kfold_splits.yaml", "kfold_splits.yml", "kfold_splits.json"):
            cand = os.path.join(path, name)
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(
            f"目录 {path} 下未找到 kfold_splits.yaml/.json；"
            f"请先运行 make_kfold_splits.py"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"划分文件不存在: {path}")
    return path


def load_splits_meta(splits_path):
    """
    加载预划分文件的 meta（兼容顶层字段或嵌套 meta 块）。
    返回 (resolved_splits_path, meta_dict, raw_data)。
    """
    path = resolve_splits_path(splits_path, required=True)
    data = load_config_file(path)
    meta = {k: v for k, v in data.items() if k != "folds"}
    if isinstance(meta.get("meta"), dict):
        nested = meta.pop("meta")
        merged = dict(nested)
        merged.update(meta)
        meta = merged
    return path, meta, data


def resolve_csv_path_from_splits(splits_path, csv_override=None):
    """
    从预划分 meta.csv_path 解析数据 CSV。
    若提供 csv_override 则优先使用（用于 CSV 搬迁后的兼容覆盖）。
    """
    if csv_override is not None and str(csv_override).strip():
        path = os.path.abspath(str(csv_override))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV 不存在: {path}")
        return path

    resolved, meta, _ = load_splits_meta(splits_path)
    csv_path = meta.get("csv_path")
    if csv_path is None or str(csv_path).strip() == "":
        raise ValueError(
            f"预划分文件缺少 meta.csv_path: {resolved}。"
            f"请用 make_kfold_splits.py 重新生成，或用 --csv_path 临时覆盖。"
        )
    path = os.path.abspath(str(csv_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"预划分记录的 CSV 不存在: {path}\n"
            f"  来源划分: {resolved}\n"
            f"  可用 --csv_path 覆盖该路径。"
        )
    return path


def resolve_runtime_splits_path(args, cfg=None):
    """解析训练用预划分路径：优先 CLI --splits_path，其次 config.splits_path。"""
    if getattr(args, "splits_path", None):
        return resolve_splits_path(args.splits_path, required=True)

    if cfg and cfg.get("splits_path"):
        return resolve_splits_path(cfg.get("splits_path"), required=True)

    raise ValueError(
        "必须提供预划分结果：请指定 --splits_path。"
        "请先运行 make_kfold_splits.py。"
    )


def load_kfold_splits(splits_path, pt):
    """
    从预划分文件加载 K 折索引。
    返回 (splits, split_meta, fold_records)，splits 为 [(train_idx, val_idx), ...]。
    """
    path, meta, data = load_splits_meta(splits_path)
    if "folds" not in data or not data["folds"]:
        raise ValueError(f"划分文件缺少 folds: {path}")

    case_to_idx = {str(c): i for i, c in enumerate(pt["case_id"].astype(str).tolist())}
    all_cases = set(case_to_idx.keys())
    splits = []
    fold_records = []

    for rec in data["folds"]:
        fold = int(rec.get("fold", len(fold_records)))
        tr_ids = [str(x) for x in rec.get("train_case_ids", [])]
        va_ids = [str(x) for x in rec.get("val_case_ids", [])]
        missing = [c for c in tr_ids + va_ids if c not in case_to_idx]
        if missing:
            raise KeyError(
                f"划分 fold={fold} 中有 {len(missing)} 个 case_id 不在当前 CSV："
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        overlap = set(tr_ids) & set(va_ids)
        if overlap:
            raise ValueError(f"fold={fold} train/val 存在重叠 case_id: {sorted(overlap)[:5]}")

        tr_idx = np.asarray([case_to_idx[c] for c in tr_ids], dtype=int)
        va_idx = np.asarray([case_to_idx[c] for c in va_ids], dtype=int)
        splits.append((tr_idx, va_idx))

        # 用当前患者表重算分布，保证与本次 CSV 一致
        rebuilt = build_fold_split_record(
            pt, fold, tr_idx, va_idx, rec.get("stratify_by", meta.get("stratify_by"))
        )
        fold_records.append(rebuilt)

    covered = set()
    for tr_idx, va_idx in splits:
        covered.update(pt.iloc[tr_idx]["case_id"].astype(str).tolist())
        covered.update(pt.iloc[va_idx]["case_id"].astype(str).tolist())
    unused = sorted(all_cases - covered)
    if unused:
        print(f"警告: 当前 CSV 中有 {len(unused)} 个 case 未出现在划分中（将被忽略）: "
              f"{unused[:5]}{'...' if len(unused) > 5 else ''}")

    meta = dict(meta)
    meta["source_splits_path"] = path
    meta["loaded_from_file"] = True
    meta["k"] = len(splits)
    print(f"已从预划分加载 K 折: {path} (k={len(splits)}, "
          f"stratify_by={meta.get('stratify_by')}, "
          f"split_type={meta.get('split_type')}, subgroup={meta.get('subgroup')})")
    return splits, meta, fold_records


def train_kfold(pt, cfg, device, log_dir, eval_overall_splits=None):
    """K 折训练：强制从 --splits_path 加载预划分，不再现场划分。"""
    splits_path = resolve_splits_path(cfg.get("splits_path"), required=True)
    cfg["splits_path"] = splits_path
    splits, split_meta, fold_records = load_kfold_splits(splits_path, pt)

    # 与预划分对齐
    cfg["k"] = int(split_meta.get("k", len(splits)))
    if split_meta.get("stratify_by"):
        cfg["stratify_by"] = split_meta["stratify_by"]
    split_meta["train_seed"] = cfg.get("seed")
    split_meta["loaded_from_file"] = True
    split_meta["source_splits_path"] = splits_path

    # 亚组专训：定位总体划分，便于在总体验证集上评估
    subgroup_mode = is_subgroup_split_meta(split_meta)
    overall_splits_path = resolve_overall_splits_path(
        splits_path, explicit=eval_overall_splits
    )
    overall_splits = overall_fold_records = overall_split_meta = None
    if overall_splits_path:
        try:
            overall_splits, overall_split_meta, overall_fold_records = load_kfold_splits(
                overall_splits_path, pt
            )
            print(f"已加载总体划分用于评估: {overall_splits_path}")
        except Exception as e:
            print(f"警告: 加载总体划分失败，将跳过总体验证评估: {e}")
            overall_splits = overall_fold_records = overall_split_meta = None
    elif subgroup_mode:
        print(
            "提示: 当前为亚组专训，但未找到父级总体划分；"
            "仅输出本亚组验证集指标。可用 --eval_overall_splits 指定总体 kfold_splits。"
        )

    # 将划分副本写入本次实验日志，保证实验自包含
    save_kfold_splits(log_dir, split_meta, fold_records)

    fold_summaries = []
    best_epochs = []
    val_aucs = []
    oof_frames = []
    overall_eval_frames = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\n========== Fold {fold} / {cfg['k']} ==========")
        print(
            f"  train Molecular: {fold_records[fold]['train_molecular']}, "
            f"val Molecular: {fold_records[fold]['val_molecular']}"
        )
        print(
            f"  train Molecular|label: {fold_records[fold]['train_molecular_label']}, "
            f"val Molecular|label: {fold_records[fold]['val_molecular_label']}"
        )
        pt_tr = pt.iloc[tr_idx].reset_index(drop=True)
        pt_va = pt.iloc[va_idx].reset_index(drop=True)
        fold_dir = os.path.join(log_dir, f"fold_{fold}")
        res = train_one_run(pt_tr, pt_va, cfg, device, fold_dir, fold_tag=f"fold{fold}")
        best_epochs.append(res["best_epoch"])
        val_aucs.append(res["best_auc"])

        fold_summary = {
            "fold": fold,
            "best_epoch": res["best_epoch"],
            "best_val_auc": res["best_auc"],
            "n_train": int(len(pt_tr)),
            "n_val": int(len(pt_va)),
            "n_train_pos": int((pt_tr["y"] == 1).sum()),
            "n_val_pos": int((pt_va["y"] == 1).sum()),
            "train_molecular": fold_records[fold]["train_molecular"],
            "val_molecular": fold_records[fold]["val_molecular"],
            "val_metrics": res.get("val_metrics"),
        }

        if res.get("val_predictions") is not None and len(res["val_predictions"]):
            pred_df = res["val_predictions"].copy()
            pred_df["fold"] = fold
            oof_frames.append(pred_df)

        # 亚组专训 / 显式总体划分：在总体 fold 验证集上再评估一次
        if overall_splits is not None and fold < len(overall_splits):
            _, ov_va_idx = overall_splits[fold]
            pt_ov = pt.iloc[ov_va_idx].reset_index(drop=True)
            ckpt_path = res.get("ckpt_best") or resolve_ckpt_path(fold_dir)
            if ckpt_path and len(pt_ov) > 0:
                encoder = res.get("encoder")
                if encoder is None:
                    enc_path = os.path.join(fold_dir, "clinical_encoder.json")
                    if os.path.isfile(enc_path):
                        encoder = load_clinical_encoder(enc_path)
                eval_model, eval_cfg = load_model_for_eval(
                    cfg, device, ckpt_path, encoder=encoder
                )
                ov_pred = predict_patient_table(
                    eval_model, pt_ov, eval_cfg, device, clinical_encoder=encoder
                )
                ov_pred["fold"] = fold
                _, ov_metrics = save_prediction_outputs(
                    fold_dir, ov_pred, prefix="val_overall",
                    extra_meta={
                        "ckpt_path": os.path.abspath(ckpt_path),
                        "eval_splits_path": os.path.abspath(overall_splits_path),
                        "split": "overall_fold_val",
                    },
                    cfg=cfg,
                )
                print_metrics_block(f"[fold{fold}] overall-val", ov_metrics)
                fold_summary["overall_val_metrics"] = ov_metrics
                overall_eval_frames.append(ov_pred)

        fold_summaries.append(fold_summary)

    n_boot, boot_seed, boot_ci = resolve_bootstrap_params(cfg=cfg)

    # OOF：各折验证集预测拼接（样本级 bootstrap CI）
    oof_metrics = None
    if oof_frames:
        oof_df = pd.concat(oof_frames, ignore_index=True)
        _, oof_metrics = save_prediction_outputs(
            log_dir, oof_df, prefix="oof",
            extra_meta={"split": "oof_val", "source_splits_path": splits_path},
            cfg=cfg,
        )
        print_metrics_block("[OOF]", oof_metrics)

    overall_oof_metrics = None
    if overall_eval_frames:
        ov_df = pd.concat(overall_eval_frames, ignore_index=True)
        _, overall_oof_metrics = save_prediction_outputs(
            log_dir, ov_df, prefix="oof_overall",
            extra_meta={
                "split": "oof_overall_val",
                "eval_splits_path": overall_splits_path,
                "subgroup_training": subgroup_mode,
            },
            cfg=cfg,
        )
        print_metrics_block("[OOF-overall]", overall_oof_metrics)

    # K 折平均：对各折验证指标做折级 bootstrap，给出均值 95% CI
    fold_val_bundles = [
        fs["val_metrics"] for fs in fold_summaries
        if isinstance(fs.get("val_metrics"), dict) and fs["val_metrics"].get("overall")
    ]
    mean_val_metrics = aggregate_fold_metrics_bootstrap(
        fold_val_bundles, n_boot=n_boot, seed=boot_seed, ci=boot_ci,
    )
    # 落盘折级平均指标
    with open(os.path.join(log_dir, "mean_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_to_builtin(mean_val_metrics), f, ensure_ascii=False, indent=2)
    save_yaml(mean_val_metrics, os.path.join(log_dir, "mean_val_metrics.yaml"))

    mean_overall_val_metrics = None
    fold_ov_bundles = [
        fs["overall_val_metrics"] for fs in fold_summaries
        if isinstance(fs.get("overall_val_metrics"), dict)
        and fs["overall_val_metrics"].get("overall")
    ]
    if fold_ov_bundles:
        mean_overall_val_metrics = aggregate_fold_metrics_bootstrap(
            fold_ov_bundles, n_boot=n_boot, seed=boot_seed, ci=boot_ci,
        )
        with open(os.path.join(log_dir, "mean_overall_val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(_to_builtin(mean_overall_val_metrics), f, ensure_ascii=False, indent=2)
        save_yaml(
            mean_overall_val_metrics,
            os.path.join(log_dir, "mean_overall_val_metrics.yaml"),
        )

    valid = [v for v in val_aucs if v is not None and v == v]
    mean_auc = mean_val_metrics.get("overall", {}).get("auc")
    summary = {
        "split_mode": "kfold",
        "k": cfg["k"],
        "modality": cfg.get("modality"),
        "clinical_only": bool(cfg.get("clinical_only", False)),
        "stratify_by": split_meta.get("stratify_by"),
        "stratify_by_requested": split_meta.get("stratify_by_requested"),
        "stratify_fallback": split_meta.get("stratify_fallback", False),
        "splits_file": "kfold_splits.yaml",
        "source_splits_path": split_meta.get("source_splits_path"),
        "loaded_from_file": split_meta.get("loaded_from_file", False),
        "subgroup_training": subgroup_mode,
        "subgroup": split_meta.get("subgroup"),
        "eval_overall_splits_path": overall_splits_path,
        "folds": fold_summaries,
        "best_epochs": best_epochs,
        "val_auc_per_fold": val_aucs,
        "mean_val_auc": float(mean_auc) if mean_auc == mean_auc else (
            float(np.mean(valid)) if valid else None
        ),
        "std_val_auc": float(np.std(valid)) if valid else None,
        "mean_val_auc_ci95_low": mean_val_metrics.get("overall", {}).get("auc_ci95_low"),
        "mean_val_auc_ci95_high": mean_val_metrics.get("overall", {}).get("auc_ci95_high"),
        "mean_val_metrics": mean_val_metrics,
        "mean_overall_val_metrics": mean_overall_val_metrics,
        "oof_metrics": oof_metrics,
        "oof_overall_metrics": overall_oof_metrics,
        "n_boot": n_boot,
        "bootstrap_ci": boot_ci,
        "label_map": {"N-pCR": 0, "pCR": 1},
    }
    save_yaml(summary, os.path.join(log_dir, "kfold_summary.yaml"))
    with open(os.path.join(log_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_to_builtin(summary), f, ensure_ascii=False, indent=2)
    print("\n===== K-fold 完成 =====")
    print(f"stratify_by: {split_meta.get('stratify_by')}")
    print(f"best_epochs: {best_epochs}")
    print(
        f"mean_val_auc: {_fmt_metric_ci(mean_val_metrics.get('overall', {}), 'auc')} "
        f"(std={summary['std_val_auc']})"
    )
    print_metrics_block("K折平均(折级bootstrap)", mean_val_metrics)
    if mean_overall_val_metrics is not None:
        print_metrics_block("K折平均-overall(折级bootstrap)", mean_overall_val_metrics)
    if oof_metrics is not None:
        print_metrics_block("最终 OOF(样本级bootstrap)", oof_metrics)
    if overall_oof_metrics is not None:
        print_metrics_block(
            "最终 OOF-overall（亚组专训/总体评估, 样本级bootstrap）",
            overall_oof_metrics,
        )
    return summary


def train_all(pt, cfg, device, log_dir):
    print("\n========== All-train（全量训练） ==========")
    res = train_one_run(pt, None, cfg, device, log_dir, fold_tag="all")
    summary = {
        "split_mode": "all_train",
        "best_epoch_lowest_loss": res["best_epoch"],
        "lowest_train_loss": res["best_loss"],
        "n_train": int(len(pt)),
        "n_pos": int((pt["y"] == 1).sum()),
        "ckpt_last": os.path.join(log_dir, "checkpoint_last.pt"),
        "ckpt_best_loss": os.path.join(log_dir, "checkpoint_best_loss.pt"),
        "label_map": {"N-pCR": 0, "pCR": 1},
    }
    save_yaml(summary, os.path.join(log_dir, "all_train_summary.yaml"))
    with open(os.path.join(log_dir, "all_train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_to_builtin(summary), f, ensure_ascii=False, indent=2)
    print("\n===== All-train 完成 =====")
    print(f"最低 train loss epoch: {res['best_epoch']}, loss={res['best_loss']:.4f}")
    return summary


# ============================================================================
# 工具
# ============================================================================
def set_seed(seed=1):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_optimizer(model, cfg):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg["opt"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["reg"])
    return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["reg"])


def read_csv_smart(path):
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def build_config(args):
    """组装超参数配置：CLI 默认 -> 若指定 --config 则用 yaml/json 覆盖 HPARAM_KEYS。"""
    cfg = {k: getattr(args, k) for k in HPARAM_KEYS if hasattr(args, k)}
    cfg["n_classes"] = 2
    if args.config and os.path.isfile(args.config):
        override = load_config_file(args.config)
        for k, v in override.items():
            if k in HPARAM_KEYS:
                cfg[k] = v
        print(f"已从 {args.config} 覆盖超参数: "
              f"{[k for k in override if k in HPARAM_KEYS]}")
    cfg["n_classes"] = 2
    return cfg


def save_run_config(cfg, log_dir):
    """将超参数以 YAML 写入日志（主格式）；额外保留 json 便于兼容。"""
    os.makedirs(log_dir, exist_ok=True)
    yaml_path = os.path.join(log_dir, "config.yaml")
    json_path = os.path.join(log_dir, "config.json")
    save_yaml(cfg, yaml_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(cfg), f, ensure_ascii=False, indent=2)
    print(f"超参数已保存到 {yaml_path}（并同步 {json_path}）")
    return yaml_path


def get_args():
    p = argparse.ArgumentParser(description="乳腺 NAC pCR 二分类训练脚本")

    # 数据 / 路径（CSV 从预划分 meta.csv_path 读取）
    p.add_argument("--log_root", type=str, default="./logs", help="日志根目录")
    p.add_argument("--exp_name", type=str, default="exp", help="实验名")
    p.add_argument("--config", type=str, default=None,
                   help="超参数配置文件（yaml/yml/json）：覆盖默认超参")

    # 划分（数据入口：必须依赖预划分）
    p.add_argument(
        "--splits_path",
        type=str,
        required=True,
        help="预划分结果路径（必填）：kfold_splits.yaml/.json 或其父目录。"
             "CSV 从该文件 meta.csv_path 读取。须先运行 make_kfold_splits.py 生成",
    )
    p.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="可选：覆盖预划分 meta.csv_path（仅当原 CSV 路径失效时使用）",
    )
    p.add_argument("--split_mode", choices=["kfold", "all_train"], default="kfold")
    p.add_argument(
        "--k", type=int, default=5,
        help="兼容字段；kfold 时以 --splits_path 文件中的 k 为准",
    )
    p.add_argument(
        "--stratify_by",
        type=str,
        default="Molecular_label",
        choices=list(STRATIFY_BY_CHOICES),
        help="兼容字段；kfold 时以预划分文件中的 stratify_by 为准。"
             "实际划分请使用 make_kfold_splits.py --stratify_by",
    )
    p.add_argument(
        "--eval_overall_splits",
        type=str,
        default=None,
        help="亚组专训时用于总体验证评估的划分（kfold_splits.yaml 或目录）。"
             "默认若 --splits_path 位于 molecular_subgroups/<Molecular>/ 下则自动上溯父级总体划分",
    )

    # 标签 / 特征路径列
    p.add_argument("--label_col", type=str, default="label", help="标签列名（N-pCR/pCR 或 0/1）")
    p.add_argument("--feat_path_col", type=str, default=None,
                   help="特征路径列名，默认自动识别 slide_feats_path / slide_feat_path")

    # 训练超参
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--reg", type=float, default=1e-5, help="weight decay")
    p.add_argument("--drop_out", type=float, default=0.25)
    p.add_argument("--gc", type=int, default=16, help="梯度累积步数")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--n_boot", type=int, default=DEFAULT_N_BOOT,
        help="终末评估 bootstrap 次数（样本级 / K折级），默认 1000；<=0 关闭",
    )
    p.add_argument(
        "--bootstrap_ci", type=float, default=DEFAULT_BOOTSTRAP_CI,
        help="bootstrap 置信水平，默认 0.95",
    )
    p.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    p.add_argument("--num_workers", type=int, default=2)

    # 早停（有验证集时生效；all_train 无 val 时自动跳过）
    p.add_argument(
        "--early_stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用早停（默认开启；--no-early_stop 关闭）",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="早停耐心：监控指标连续多少个 epoch 无提升则停止（默认 10）",
    )
    p.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="判定为提升的最小变化量（默认 0；val_auc 等越大越好，val_loss 越小越好）",
    )
    p.add_argument(
        "--early_stop_metric",
        type=str,
        default="val_auc",
        choices=list(EARLY_STOP_METRIC_CHOICES),
        help="早停监控指标（默认 val_auc）："
             "val_auc/val_auprc/val_loss/val_acc/val_balanced_acc/"
             "val_f1/val_precision/val_recall/val_sensitivity/val_specificity/val_mcc",
    )

    # 模型
    p.add_argument("--model_type",
                   choices=["abmil", "mean_mil", "max_mil", "mamba_mil", "trans_mil", "s4model"],
                   default="abmil")
    p.add_argument("--in_dim", type=int, default=-1, help="特征维度，<=0 时从特征文件自动推断")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--max_slides_train", type=int, default=3,
                   help="训练时单患者最多拼接的 slide 数，超过则随机采样")
    p.add_argument("--feat_key", type=str, default="features",
                   help="h5/dict 中特征键名（默认 features）；.pt 为 tensor 时可忽略")

    # 模态 / 临床特征
    p.add_argument(
        "--modality",
        type=str,
        default="pathomic",
        choices=list(MODALITY_CHOICES),
        help="输入模态：pathomic=病理+临床融合（默认）；pathology=仅病理；clinical=仅临床",
    )
    p.add_argument(
        "--clinical_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅使用临床信息预测（等价于 --modality clinical）",
    )
    p.add_argument("--use_clinical", action=argparse.BooleanOptionalAction, default=True,
                   help="兼容旧开关：--no-use_clinical 等价于 --modality pathology；"
                        "在 pathomic 下是否融合临床（由 modality 最终决定）")
    p.add_argument("--fusion_type", choices=["concat", "bilinear", "gated"], default="concat",
                   help="MIL 全局表征与临床嵌入的中期融合方式（仅 pathomic）")
    p.add_argument("--clinical_hidden_dim", type=int, default=256,
                   help="临床 MLP 隐藏维；clinical_only 时作为分类器宽度，"
                        "pathomic 时嵌入维 = max(32, hidden_dim//2)")

    # MambaMIL 专用
    p.add_argument("--mambamil_layer", type=int, default=2)
    p.add_argument("--mambamil_rate", type=int, default=10)
    p.add_argument("--mambamil_type", choices=["Mamba", "BiMamba", "SRMamba"], default="SRMamba")

    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    cfg = build_config(args)

    # 统一模态（clinical_only / use_clinical 别名）
    if getattr(args, "clinical_only", False):
        cfg["clinical_only"] = True
    if args.config is None and not cfg.get("clinical_only", False):
        # CLI --no-use_clinical 时，若未显式指定其他 modality，则视为 pathology
        if cfg.get("use_clinical") is False and cfg.get("modality", "pathomic") == "pathomic":
            cfg["modality"] = "pathology"
    modality = normalize_modality(cfg)

    # 必须依赖预划分；CSV 从 meta.csv_path 读取
    set_seed(cfg["seed"])
    splits_path = resolve_runtime_splits_path(args, cfg)
    csv_path = resolve_csv_path_from_splits(
        splits_path, csv_override=getattr(args, "csv_path", None)
    )
    cfg["splits_path"] = splits_path
    cfg["csv_path"] = csv_path
    print(f"数据来自预划分: splits={splits_path}")
    print(f"CSV: {csv_path}")

    df = read_csv_smart(csv_path)
    pt, clinical_cols, feat_col = build_patient_table(
        df,
        label_col=cfg.get("label_col", "label"),
        feat_path_col=cfg.get("feat_path_col"),
        require_feats=(modality != "clinical"),
    )
    cfg["feat_path_col"] = feat_col
    cfg["n_classes"] = 2

    if modality == "clinical":
        cfg["in_dim"] = 0
    elif cfg["in_dim"] is None or cfg["in_dim"] <= 0:
        all_paths = [p for ps in pt["feat_paths"] for p in ps]
        cfg["in_dim"] = detect_in_dim(all_paths, cfg["feat_key"])

    print(
        f"患者数: {len(pt)}, pCR(1)={int((pt['y'] == 1).sum())}, "
        f"N-pCR(0)={int((pt['y'] == 0).sum())}, "
        f"modality={modality}, in_dim={cfg['in_dim']}, n_classes={cfg['n_classes']}, "
        f"临床列: {clinical_cols}, fusion: {cfg.get('fusion_type', 'concat')}, "
        f"split_mode: {cfg.get('split_mode')}, "
        f"splits_path: {cfg.get('splits_path')}"
    )

    log_dir = os.path.join(args.log_root, args.exp_name)
    save_run_config(cfg, log_dir)

    if cfg["split_mode"] == "kfold":
        train_kfold(
            pt, cfg, device, log_dir,
            eval_overall_splits=getattr(args, "eval_overall_splits", None),
        )
    else:
        # all_train 仍要求预划分（用于定位 CSV），但不按折训练
        train_all(pt, cfg, device, log_dir)


if __name__ == "__main__":
    main()
