#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_pcr.py
===========
基于 BreastRCB-Prognosis 中期融合框架改造的乳腺新辅助治疗 pCR 二分类单脚本。

任务：
  * 标签映射：N-pCR -> 0，pCR -> 1
  * 损失：CrossEntropy
  * 评价：AUC / Accuracy / F1 / Sensitivity / Specificity
  * 一个患者(case)可能有多张 slide，训练时拼接成一个 bag；若 slide 数 > max_slides
    则随机选取 max_slides 张拼接；推理时拼接全部 slide。

输入 CSV 格式见项目根目录 example_dataset.csv，关键列：
  case_id, slide_id, slide_feats_path, label,
  Molecular, T, N, Age, ER, PR, HER2, Ki67
其中 label 可为字符串 N-pCR/pCR，或已映射的 0/1。
slide_feats_path 指向每张 slide 的特征文件（.pt 或 .h5）。
临床融合白名单列：
  因子变量（one-hot）：Molecular, T, N, HER2
  连续变量（标准化）：Age, ER, PR, Ki67
Molecular 取值（四种）：HR+HER2- / HR+HER2+ / TNBC / HER2。

支持三种运行模式（--mode）:
  train  : 训练。--split_mode 控制 kfold（默认）或 all_train。
  infer  : 推理。给定超参 yaml/json + 权重 + csv，输出指标和每个患者的预测概率。

K 折划分：由 --stratify_by 指定分层依据（默认 Molecular_label，即
  label 与 Molecular 联合分层）；划分结果写入日志
  （kfold_splits.yaml 与各 fold_*/split.yaml）。
超参数：训练时以 config.yaml 写入日志目录；推理 --config 支持 yaml/json。

示例（在项目根目录执行）：
  python main_pcr.py --mode train --split_mode kfold \\
      --csv_path example_dataset.csv --log_root ./logs --exp_name pcr_kfold

  python main_pcr.py --mode train --split_mode all_train \\
      --csv_path example_dataset.csv --log_root ./logs --exp_name pcr_all

  python main_pcr.py --mode infer --config ./logs/pcr_kfold/config.yaml \\
      --ckpt_path ./logs/pcr_kfold/fold_0/checkpoint_best.pt \\
      --csv_path test.csv --save_infer_dir ./infer_pcr
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
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, KFold

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
    "k", "split_mode", "stratify_by", "n_classes", "max_epochs", "lr", "reg",
    "drop_out", "gc", "seed", "opt", "model_type", "in_dim",
    "hidden_dim", "max_slides_train", "feat_key", "num_workers",
    "mambamil_layer", "mambamil_rate", "mambamil_type",
    "use_clinical", "fusion_type", "clinical_hidden_dim", "clinical_in_dim",
    "label_col", "feat_path_col",
]

# K 折分层依据（写入 config.yaml / kfold_splits.yaml 的 stratify_by）
STRATIFY_BY_CHOICES = ("Molecular_label", "Molecular", "label", "none")

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
        if cfg.get("use_clinical", True):
            raise NotImplementedError(
                f"临床中期融合暂不支持 model_type={mt}，请使用 abmil/mean_mil/max_mil，"
                f"或设置 use_clinical=false"
            )
        return _build_repo_model(cfg)
    raise NotImplementedError(f"未知 model_type: {mt}")


def build_model(cfg, device):
    if cfg["model_type"] in ("mamba_mil", "trans_mil", "s4model") and not cfg.get("use_clinical", True):
        model = build_backbone(cfg)
    else:
        backbone = build_backbone(cfg)
        clinical_in_dim = int(cfg.get("clinical_in_dim", 0) or 0)
        use_clinical = bool(cfg.get("use_clinical", True)) and clinical_in_dim > 0
        model = PathomicClassificationModel(
            backbone, cfg["n_classes"], clinical_in_dim,
            fusion_type=cfg.get("fusion_type", "concat"),
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["drop_out"],
            use_clinical=use_clinical,
        )
    return model.to(device)


def _build_repo_model(cfg):
    """从 MambaMIL 仓库动态构建模型（仅 path-only，无临床融合）。"""
    import sys

    # 可选依赖：与本脚本同级的 MambaMIL 目录
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MambaMIL")
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    mt = cfg["model_type"]

    class RepoClsWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

        def forward(self, x, clinical=None):
            out = self.model(x)
            # 兼容 (logits, ...) 或纯 logits
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    try:
        if mt == "mamba_mil":
            from models.MambaMIL import MambaMIL
            m = MambaMIL(in_dim=cfg["in_dim"], n_classes=cfg["n_classes"],
                         dropout=cfg["drop_out"], act="gelu", survival=False,
                         layer=cfg["mambamil_layer"], rate=cfg["mambamil_rate"],
                         type=cfg["mambamil_type"])
        elif mt == "trans_mil":
            from models.TransMIL import TransMIL
            m = TransMIL(cfg["in_dim"], cfg["n_classes"], dropout=cfg["drop_out"],
                         act="relu", survival=False)
        else:
            from models.S4MIL import S4Model
            m = S4Model(in_dim=cfg["in_dim"], n_classes=cfg["n_classes"], act="gelu",
                        dropout=cfg["drop_out"], survival=False)
    except Exception as e:
        raise RuntimeError(
            f"无法从 MambaMIL 仓库加载模型 '{mt}'（可能缺少依赖）。"
            f"可改用 model_type=abmil。原始错误: {e}"
        )
    return RepoClsWrapper(m)


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


def build_patient_table(df, label_col="label", feat_path_col=None):
    """构建患者级表：每个 case 一行，含 y / feat_paths / 临床列。"""
    if label_col not in df.columns:
        raise KeyError(f"CSV 缺少标签列 {label_col}")
    feat_path_col = resolve_feat_path_col(df, feat_path_col)

    df = df.copy()
    df["y"] = df[label_col].map(map_label)
    df = df.dropna(subset=["case_id", feat_path_col, "y"])
    clinical_cols = get_clinical_columns(df)

    records = []
    for case_id, g in df.groupby("case_id", sort=False):
        labels = g["y"].astype(int).tolist()
        if len(set(labels)) != 1:
            raise ValueError(f"同一 case_id={case_id} 存在不一致 label: {labels}")
        feat_paths = list(g[feat_path_col].astype(str))
        rec = {
            "case_id": case_id,
            "y": int(labels[0]),
            "feat_paths": feat_paths,
        }
        for col in clinical_cols:
            rec[col] = g[col].iloc[0]
        records.append(rec)
    pt = pd.DataFrame(records).reset_index(drop=True)
    return pt, clinical_cols, feat_path_col


class PCRBagDataset(torch.utils.data.Dataset):
    """患者级数据集；每个样本返回 bag 特征 + 临床向量 + 二分类标签。"""

    def __init__(self, pt_df, feat_key, max_slides_train, training, clinical_encoder=None):
        self.pt = pt_df.reset_index(drop=True)
        self.feat_key = feat_key
        self.max_slides_train = max_slides_train
        self.training = training
        self.clinical_encoder = clinical_encoder
        if clinical_encoder is not None and clinical_encoder.output_dim > 0:
            self.clinical_matrix = clinical_encoder.transform_df(self.pt)
        else:
            self.clinical_matrix = None

    def __len__(self):
        return len(self.pt)

    def __getitem__(self, idx):
        row = self.pt.iloc[idx]
        paths = list(row["feat_paths"])
        if self.training and self.max_slides_train > 0 and len(paths) > self.max_slides_train:
            paths = random.sample(paths, self.max_slides_train)
        feats = [load_features(p, self.feat_key) for p in paths]
        feats = np.concatenate(feats, axis=0)
        if self.clinical_matrix is not None:
            clinical = torch.from_numpy(self.clinical_matrix[idx]).float()
        else:
            clinical = torch.zeros((0,), dtype=torch.float32)
        return (
            torch.from_numpy(feats).float(),
            clinical,
            int(row["y"]),
            str(row["case_id"]),
        )


def collate_bag(batch):
    feats, clinical, label, cid = batch[0]
    return feats, clinical, label, cid


def make_loader(pt_df, cfg, training, clinical_encoder=None):
    ds = PCRBagDataset(
        pt_df, cfg["feat_key"], cfg["max_slides_train"], training, clinical_encoder
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=training, num_workers=cfg["num_workers"],
        collate_fn=collate_bag,
    )


def prepare_clinical_encoder(pt_train, cfg, out_dir=None):
    if not cfg.get("use_clinical", True):
        cfg["clinical_in_dim"] = 0
        return None
    encoder = ClinicalEncoder().fit(pt_train)
    cfg["clinical_in_dim"] = int(encoder.output_dim)
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
    use_clinical = bool(cfg.get("use_clinical", True)) and int(cfg.get("clinical_in_dim", 0) or 0) > 0
    if use_clinical:
        clinical = clinical.to(device, non_blocking=True)
        return model(feats, clinical)
    return model(feats, None)


# ============================================================================
# 指标
# ============================================================================
def compute_cls_metrics(y_true, y_prob, y_pred=None, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_pred is None:
        y_pred = (y_prob >= thr).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
        "acc": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else float("nan"),
    }
    # AUC 需要两类都存在
    if len(np.unique(y_true)) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = float("nan")

    if len(y_true):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    else:
        metrics["sensitivity"] = float("nan")
        metrics["specificity"] = float("nan")
    return metrics


# ============================================================================
# 训练 / 验证 / 推理
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

    for epoch in range(cfg["max_epochs"]):
        tr = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        rec = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_auc": tr["auc"],
            "train_acc": tr["acc"],
            "train_f1": tr["f1"],
        }
        if val_loader is not None:
            va = run_epoch(model, val_loader, optimizer, cfg, device, train=False)
            rec.update({
                "val_loss": va["loss"],
                "val_auc": va["auc"],
                "val_acc": va["acc"],
                "val_f1": va["f1"],
                "val_sensitivity": va["sensitivity"],
                "val_specificity": va["specificity"],
            })
            # 优先按 val AUC 选模；AUC 不可用时回退到 val ACC
            score = va["auc"] if va["auc"] == va["auc"] else va["acc"]
            if score == score and score > best_auc:
                best_auc, best_epoch = float(score), epoch
                torch.save(model.state_dict(), ckpt_best)
            print(
                f"[{fold_tag} epoch {epoch}] "
                f"train_loss={tr['loss']:.4f} train_auc={tr['auc']:.4f} "
                f"val_loss={va['loss']:.4f} val_auc={va['auc']:.4f} val_acc={va['acc']:.4f}"
            )
        else:
            if tr["loss"] < best_loss:
                best_loss, best_epoch = tr["loss"], epoch
                torch.save(model.state_dict(), ckpt_best_loss)
            print(
                f"[{fold_tag} epoch {epoch}] "
                f"train_loss={tr['loss']:.4f} train_auc={tr['auc']:.4f} train_acc={tr['acc']:.4f}"
            )

        history.append(rec)

    torch.save(model.state_dict(), ckpt_last)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_auc": best_auc if val_loader is not None else None,
        "best_loss": best_loss if val_loader is None else None,
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


def train_kfold(pt, cfg, device, log_dir):
    splits, split_meta = make_kfold_splits(pt, cfg)

    # 先落盘划分结果，再开始训练（便于中断后复现）
    fold_records = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        fold_records.append(
            build_fold_split_record(pt, fold, tr_idx, va_idx, split_meta["stratify_by"])
        )
    save_kfold_splits(log_dir, split_meta, fold_records)

    fold_summaries = []
    best_epochs = []
    val_aucs = []

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
        fold_summaries.append({
            "fold": fold,
            "best_epoch": res["best_epoch"],
            "best_val_auc": res["best_auc"],
            "n_train": int(len(pt_tr)),
            "n_val": int(len(pt_va)),
            "n_train_pos": int((pt_tr["y"] == 1).sum()),
            "n_val_pos": int((pt_va["y"] == 1).sum()),
            "train_molecular": fold_records[fold]["train_molecular"],
            "val_molecular": fold_records[fold]["val_molecular"],
        })

    valid = [v for v in val_aucs if v is not None and v == v]
    summary = {
        "split_mode": "kfold",
        "k": cfg["k"],
        "stratify_by": split_meta["stratify_by"],
        "stratify_by_requested": split_meta.get("stratify_by_requested"),
        "stratify_fallback": split_meta.get("stratify_fallback", False),
        "splits_file": "kfold_splits.yaml",
        "folds": fold_summaries,
        "best_epochs": best_epochs,
        "val_auc_per_fold": val_aucs,
        "mean_val_auc": float(np.mean(valid)) if valid else None,
        "std_val_auc": float(np.std(valid)) if valid else None,
        "label_map": {"N-pCR": 0, "pCR": 1},
    }
    save_yaml(summary, os.path.join(log_dir, "kfold_summary.yaml"))
    with open(os.path.join(log_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_to_builtin(summary), f, ensure_ascii=False, indent=2)
    print("\n===== K-fold 完成 =====")
    print(f"stratify_by: {split_meta['stratify_by']}")
    print(f"best_epochs: {best_epochs}")
    print(f"mean_val_auc: {summary['mean_val_auc']}")
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


@torch.no_grad()
def run_inference(cfg, device, args):
    df = read_csv_smart(args.csv_path)
    pt, _, _ = build_patient_table(
        df,
        label_col=cfg.get("label_col", "label"),
        feat_path_col=cfg.get("feat_path_col"),
    )

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
    encoder_path = os.path.join(ckpt_dir, "clinical_encoder.json")
    encoder = None
    if cfg.get("use_clinical", True):
        if os.path.isfile(encoder_path):
            encoder = load_clinical_encoder(encoder_path)
            cfg["clinical_in_dim"] = encoder.output_dim
        else:
            print(f"警告: 未找到 {encoder_path}，将不使用临床特征推理")
            cfg["use_clinical"] = False
            cfg["clinical_in_dim"] = 0

    model = build_model(cfg, device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    loader = make_loader(pt, cfg, training=False, clinical_encoder=encoder)
    rows, ys, probs, preds = [], [], [], []
    for feats, clinical, label, cid in loader:
        feats = feats.to(device)
        logits = model_forward(model, feats, clinical, cfg, device)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        prob1 = float(torch.softmax(logits, dim=-1)[0, 1].cpu().item())
        pred = int(torch.argmax(logits, dim=-1).cpu().item())
        rows.append({
            "case_id": cid,
            "label": int(label),
            "prob_pCR": prob1,
            "pred": pred,
        })
        ys.append(int(label))
        probs.append(prob1)
        preds.append(pred)

    os.makedirs(args.save_infer_dir, exist_ok=True)
    pred_df = pd.DataFrame(rows)
    pred_csv = os.path.join(args.save_infer_dir, "predictions.csv")
    pred_df.to_csv(pred_csv, index=False)

    metrics = compute_cls_metrics(ys, probs, preds)
    metrics.update({
        "ckpt_path": os.path.abspath(args.ckpt_path),
        "csv_path": os.path.abspath(args.csv_path),
        "label_map": {"N-pCR": 0, "pCR": 1},
    })
    with open(os.path.join(args.save_infer_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(
        f"推理完成：AUC={metrics['auc']:.4f} ACC={metrics['acc']:.4f} F1={metrics['f1']:.4f}，"
        f"结果保存到 {args.save_infer_dir}"
    )
    print(f"  - 每患者预测: {pred_csv}")
    return metrics


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
    p = argparse.ArgumentParser(description="乳腺 NAC pCR 二分类训练/推理脚本")
    p.add_argument("--mode", choices=["train", "infer"], default="train")

    # 数据 / 路径
    p.add_argument("--csv_path", type=str, required=True, help="输入 CSV（见 example_dataset.csv）")
    p.add_argument("--log_root", type=str, default="./logs", help="日志根目录（train）")
    p.add_argument("--exp_name", type=str, default="exp", help="实验名（train）")
    p.add_argument("--config", type=str, default=None,
                   help="超参数配置文件（yaml/yml/json）：覆盖默认或提供推理所需模型超参")

    # 推理
    p.add_argument("--ckpt_path", type=str, default=None, help="权重路径（infer）")
    p.add_argument("--save_infer_dir", type=str, default=None, help="推理结果保存目录（infer）")

    # 划分
    p.add_argument("--split_mode", choices=["kfold", "all_train"], default="kfold")
    p.add_argument("--k", type=int, default=5, help="k 折数量（默认 5）")
    p.add_argument(
        "--stratify_by",
        type=str,
        default="Molecular_label",
        choices=list(STRATIFY_BY_CHOICES),
        help="K 折分层依据：Molecular_label=按 Molecular 与 label 联合分层（默认）；"
             "Molecular=仅分子分型；label=仅结局标签；none=不分层随机划分。"
             "该值会写入 config.yaml / kfold_splits.yaml",
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
    p.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    p.add_argument("--num_workers", type=int, default=2)

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

    # 临床特征中期融合
    p.add_argument("--use_clinical", action=argparse.BooleanOptionalAction, default=True,
                   help="是否使用临床白名单列做中期融合（默认开启；含 Molecular；"
                        "因子变量 Molecular/T/N/HER2 one-hot，连续变量 Age/ER/PR/Ki67 标准化）")
    p.add_argument("--fusion_type", choices=["concat", "bilinear", "gated"], default="concat",
                   help="MIL 全局表征与临床嵌入的中期融合方式")
    p.add_argument("--clinical_hidden_dim", type=int, default=256,
                   help="临床 MLP 中间层维度（实际嵌入维 = max(32, hidden_dim//2)）")

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

    if args.mode == "infer":
        assert args.ckpt_path and args.save_infer_dir, "推理需要 --ckpt_path 与 --save_infer_dir"
        assert args.config, "推理需要 --config 指定训练时保存的超参数 yaml/json"
        if cfg.get("in_dim", -1) is None or cfg.get("in_dim", -1) <= 0:
            df = read_csv_smart(args.csv_path)
            feat_col = resolve_feat_path_col(df, cfg.get("feat_path_col"))
            cfg["in_dim"] = detect_in_dim(df[feat_col].astype(str).tolist(), cfg["feat_key"])
        set_seed(cfg["seed"])
        run_inference(cfg, device, args)
        return

    # ---- 训练 ----
    set_seed(cfg["seed"])
    df = read_csv_smart(args.csv_path)
    pt, clinical_cols, feat_col = build_patient_table(
        df, label_col=cfg.get("label_col", "label"), feat_path_col=cfg.get("feat_path_col")
    )
    cfg["feat_path_col"] = feat_col
    cfg["n_classes"] = 2
    cfg["stratify_by"] = normalize_stratify_by(cfg.get("stratify_by", "Molecular_label"))
    if cfg["in_dim"] is None or cfg["in_dim"] <= 0:
        all_paths = [p for ps in pt["feat_paths"] for p in ps]
        cfg["in_dim"] = detect_in_dim(all_paths, cfg["feat_key"])
    print(
        f"患者数: {len(pt)}, pCR(1)={int((pt['y'] == 1).sum())}, "
        f"N-pCR(0)={int((pt['y'] == 0).sum())}, "
        f"特征维度: {cfg['in_dim']}, n_classes: {cfg['n_classes']}, "
        f"临床列: {clinical_cols}, fusion: {cfg.get('fusion_type', 'concat')}, "
        f"stratify_by: {cfg.get('stratify_by')}"
    )

    log_dir = os.path.join(args.log_root, args.exp_name)
    save_run_config(cfg, log_dir)

    if cfg["split_mode"] == "kfold":
        train_kfold(pt, cfg, device, log_dir)
    else:
        train_all(pt, cfg, device, log_dir)


if __name__ == "__main__":
    main()
