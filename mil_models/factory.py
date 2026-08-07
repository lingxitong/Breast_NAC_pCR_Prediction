#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 MIL_BASELINE 中的 AMD / WIKG / GDF 适配为本项目的特征 backbone。

约定：输入 [N, in_dim] 或 [1, N, in_dim]，输出 bag 表征 [1, H]，并设置 hidden_dim=H。
分类头由 PathomicClassificationModel 负责（可接临床中期融合）。
"""

from __future__ import print_function

import torch.nn as nn


BASELINE_MODEL_TYPES = ("amd_mil", "wikg_mil", "gdf_mil")


def get_act(act):
    if act is None:
        return nn.ReLU()
    if isinstance(act, nn.Module):
        return act
    key = str(act).strip().lower().replace("_", "").replace("-", "")
    mapping = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "leakyrelu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
    }
    if key not in mapping:
        raise ValueError(f"未知激活函数: {act!r}")
    return mapping[key]


class BaselineMILBackbone(nn.Module):
    """截掉 baseline 模型分类头，仅返回 WSI_feature 作为 bag 表征。"""

    def __init__(self, inner, hidden_dim):
        super().__init__()
        self.model = inner
        self.hidden_dim = int(hidden_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        out = self.model(x.float(), return_WSI_feature=True)
        h = out["WSI_feature"]
        if h.dim() == 1:
            h = h.unsqueeze(0)
        elif h.dim() == 2 and h.size(0) != 1:
            # 少数实现可能返回 [B, H]，本项目 batch_size=1
            h = h[:1]
        return h


def build_baseline_backbone(cfg):
    """
    按 cfg['model_type'] 构建 AMD/WIKG/GDF backbone。
    依赖：
      - amd_mil: einops
      - wikg_mil / gdf_mil: torch_geometric
    """
    mt = str(cfg["model_type"]).strip().lower()
    in_dim = int(cfg["in_dim"])
    n_classes = int(cfg.get("n_classes", 2))
    dropout = float(cfg.get("drop_out", 0.25))
    hidden_dim = int(cfg.get("hidden_dim", 512))

    if mt == "amd_mil":
        from .AMD_MIL import AMD_MIL

        embed_dim = int(cfg.get("amd_embed_dim") or hidden_dim)
        agent_num = int(cfg.get("amd_agent_num", 256))
        act = get_act(cfg.get("amd_act", "relu"))
        inner = AMD_MIL(
            num_classes=n_classes,
            in_dim=in_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            act=act,
            agent_num=agent_num,
        )
        return BaselineMILBackbone(inner, embed_dim)

    if mt == "wikg_mil":
        from .WIKG_MIL import WIKG_MIL

        dim_hidden = int(cfg.get("wikg_dim_hidden") or hidden_dim)
        topk = int(cfg.get("wikg_topk", 6))
        agg_type = cfg.get("wikg_agg_type", "bi-interaction")
        pool = cfg.get("wikg_pool", "attn")
        act = get_act(cfg.get("wikg_act", "LeakyReLU"))
        try:
            inner = WIKG_MIL(
                in_dim=in_dim,
                act=act,
                dim_hidden=dim_hidden,
                topk=topk,
                num_classes=n_classes,
                agg_type=agg_type,
                dropout=dropout,
                pool=pool,
            )
        except ImportError as e:
            raise ImportError(
                "wikg_mil 需要安装 torch_geometric。请执行: pip install torch_geometric"
            ) from e
        return BaselineMILBackbone(inner, dim_hidden)

    if mt == "gdf_mil":
        from .GDF_MIL import GDF_MIL

        hid_dim = int(cfg.get("gdf_hid_dim", 256))
        out_dim = int(cfg.get("gdf_out_dim", 128))
        k_components = int(cfg.get("gdf_k_components", 10))
        k_neighbors = int(cfg.get("gdf_k_neighbors", 10))
        act = str(cfg.get("gdf_act", "leaky_relu"))
        try:
            inner = GDF_MIL(
                in_dim=in_dim,
                num_classes=n_classes,
                hid_dim=hid_dim,
                out_dim=out_dim,
                k_components=k_components,
                k_neighbors=k_neighbors,
                dropout=dropout,
                lambda_smooth=float(cfg.get("gdf_lambda_smooth", 0.0)),
                lambda_nce=float(cfg.get("gdf_lambda_nce", 0.0)),
                act=act,
            )
        except ImportError as e:
            raise ImportError(
                "gdf_mil 需要安装 torch_geometric。请执行: pip install torch_geometric"
            ) from e
        return BaselineMILBackbone(inner, out_dim)

    raise NotImplementedError(f"未知 baseline model_type: {mt}")
