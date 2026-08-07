#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Subtype-Decoupled MIL (SDMIL)

核心设计：在共享 MIL bag 表征之上，显式解耦
  - 预后投影 z（供 Global / Intra 对比，学跨分型通用 + 分型内特异的化疗敏感性）
  - 分型原型 C_k（供 Inter 损失，保留分子分型生物学差异）
  - 分类头（CE，可接临床中期融合）
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class ProjectionHead(nn.Module):
    """两层 MLP + L2 normalize，输出对比表征 z。"""

    def __init__(self, in_dim, proj_dim=128, hidden_dim=None):
        super().__init__()
        h = int(hidden_dim) if hidden_dim is not None else int(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, proj_dim),
        )
        self.apply(_init_weights)

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


class GatedAttentionMIL(nn.Module):
    """本地 Gated-Attention MIL backbone，输出 [1, hidden]。"""

    def __init__(self, in_dim, dropout=0.25, hidden=512, att_dim=256):
        super().__init__()
        self.hidden_dim = int(hidden)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(self.hidden_dim, att_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.hidden_dim, att_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(att_dim, 1)
        self.apply(_init_weights)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(0)
        h = self.fc(x)
        A = self.attention_w(self.attention_V(h) * self.attention_U(h))
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        return torch.mm(A, h)


class SubtypeDecoupledMIL(nn.Module):
    """
    SDMIL 完整模型（病理 / pathomic）。

    forward(..., return_dict=False) -> logits
    forward(..., return_dict=True)  -> dict(logits, z, bag_repr, prototypes, fused)
    """

    def __init__(
        self,
        backbone,
        n_classes,
        clinical_in_dim=0,
        fusion_type="concat",
        hidden_dim=512,
        dropout=0.25,
        use_clinical=True,
        proj_dim=128,
        n_subtypes=4,
        build_fusion_fn=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_clinical = bool(use_clinical) and int(clinical_in_dim) > 0
        self.proj_dim = int(proj_dim)
        self.n_subtypes = int(n_subtypes)
        mil_dim = int(backbone.hidden_dim)

        # 对比头挂在「病理 bag 表征」上，避免临床 one-hot 直接泄漏分型信息到 z
        self.projector = ProjectionHead(mil_dim, proj_dim=self.proj_dim, hidden_dim=mil_dim)

        # 可学习分型原型 C_k
        self.prototypes = nn.Parameter(torch.randn(self.n_subtypes, self.proj_dim) * 0.05)

        if self.use_clinical:
            if build_fusion_fn is None:
                raise ValueError("use_clinical=True 时需要提供 build_fusion_fn")
            clin_emb_dim = max(32, int(hidden_dim) // 2)
            self.clinical_mlp = nn.Sequential(
                nn.Linear(int(clinical_in_dim), clin_emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fusion = build_fusion_fn(
                fusion_type, mil_dim, clin_emb_dim, int(hidden_dim), dropout
            )
            self.classifier = nn.Linear(int(hidden_dim), int(n_classes))
        else:
            self.classifier = nn.Linear(mil_dim, int(n_classes))

        self.apply(_init_weights)
        # 原型单独用较小方差初始化（apply 不会覆盖 Parameter）
        with torch.no_grad():
            nn.init.xavier_normal_(self.prototypes)

    def forward(self, x, clinical=None, return_dict=False):
        bag_repr = self.backbone(x)  # [1, mil_dim]
        z = self.projector(bag_repr)  # [1, proj_dim] L2-normalized
        proto = F.normalize(self.prototypes, dim=-1)

        if self.use_clinical:
            if clinical is None:
                raise ValueError("use_clinical=True 时必须提供 clinical 特征")
            clin = clinical.unsqueeze(0) if clinical.dim() == 1 else clinical
            clin_emb = self.clinical_mlp(clin)
            fused = self.fusion(bag_repr, clin_emb)
            logits = self.classifier(fused)
        else:
            fused = bag_repr
            logits = self.classifier(bag_repr)

        if not return_dict:
            return logits
        return {
            "logits": logits,
            "z": z,
            "bag_repr": bag_repr,
            "fused": fused,
            "prototypes": proto,
        }


def build_sdmil_backbone(cfg):
    """构建 SDMIL 默认 attention backbone（也可由外部传入其他 backbone）。"""
    in_dim = int(cfg["in_dim"])
    hidden = int(cfg.get("hidden_dim", 512))
    dropout = float(cfg.get("drop_out", 0.25))
    return GatedAttentionMIL(in_dim, dropout=dropout, hidden=hidden)
