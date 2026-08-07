#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三重对比损失（Triple-Level Loss）

解耦「分型特征」与「预后特征」：
  L_Global : 跨分型 pCR 不变性（SupCon，正样本=同 pCR 标签，忽略分型）
  L_Intra  : 分型内特异性（同分子分型内拉近同标签、推开反标签）
  L_Inter  : 跨分型排斥（样本对齐自身分型原型、远离其他分型原型）

因本项目 DataLoader batch_size=1，对比项依赖 FeatureMemoryBank 中的历史表征。
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMemoryBank(nn.Module):
    """FIFO 特征队列：缓存 (z, y, subtype_id)，供 batch_size=1 时做对比。"""

    def __init__(self, size, dim):
        super().__init__()
        size = max(1, int(size))
        dim = int(dim)
        self.register_buffer("z", torch.zeros(size, dim), persistent=False)
        self.register_buffer("y", torch.full((size,), -1, dtype=torch.long), persistent=False)
        self.register_buffer(
            "subtype", torch.full((size,), -1, dtype=torch.long), persistent=False
        )
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("filled", torch.zeros((), dtype=torch.long), persistent=False)
        self.size = size
        self.dim = dim

    @torch.no_grad()
    def reset(self):
        self.z.zero_()
        self.y.fill_(-1)
        self.subtype.fill_(-1)
        self.ptr.zero_()
        self.filled.zero_()

    @torch.no_grad()
    def update(self, z, y, subtype):
        """写入一条样本；z 应为已 L2 归一化的 [D] 或 [1, D]。"""
        if z is None:
            return
        z = z.detach()
        if z.dim() == 2:
            z = z.squeeze(0)
        if z.numel() == 0:
            return
        if z.shape[-1] != self.dim:
            # 投影维变化时跳过，避免 silent shape bug
            return
        idx = int(self.ptr.item())
        self.z[idx] = z
        self.y[idx] = int(y) if not torch.is_tensor(y) else int(y.item())
        self.subtype[idx] = (
            int(subtype) if not torch.is_tensor(subtype) else int(subtype.item())
        )
        self.ptr.fill_((idx + 1) % self.size)
        self.filled.fill_(min(self.size, int(self.filled.item()) + 1))

    def get(self):
        n = int(self.filled.item())
        if n <= 0:
            return None, None, None
        return self.z[:n], self.y[:n], self.subtype[:n]


def _as_1d_long(v, device):
    if torch.is_tensor(v):
        t = v.to(device=device, dtype=torch.long).view(-1)
    else:
        t = torch.tensor([int(v)], device=device, dtype=torch.long)
    return t


def supervised_contrastive_loss(anchor_z, anchor_y, bank_z, bank_y, temperature=0.07):
    """
    L_Global：监督对比（同 pCR 标签为正，忽略分子分型）。
    anchor_z: [1, D] 或 [D]；bank_*: [M, ...]
    """
    if bank_z is None or bank_z.numel() == 0:
        return anchor_z.new_zeros(())
    z_i = F.normalize(anchor_z.view(1, -1), dim=-1)
    z_b = F.normalize(bank_z, dim=-1)
    y_i = int(_as_1d_long(anchor_y, z_i.device)[0].item())
    y_b = bank_y.to(z_i.device).long().view(-1)
    pos_mask = y_b == y_i
    if not bool(pos_mask.any()):
        return z_i.new_zeros(())

    logits = (z_i @ z_b.t()).view(-1) / float(temperature)  # [M]
    # log_softmax over all bank keys；仅对正样本取平均
    log_prob = logits - torch.logsumexp(logits, dim=0)
    loss = -log_prob[pos_mask].mean()
    return loss


def intra_subtype_contrastive_loss(
    anchor_z, anchor_y, anchor_subtype,
    bank_z, bank_y, bank_subtype,
    temperature=0.07,
):
    """
    L_Intra：仅在同一分子分型内做对比。
    正样本：同 subtype 且同 y；分母：同 subtype 全部样本（含难例负样本）。
    """
    if bank_z is None or bank_z.numel() == 0:
        return anchor_z.new_zeros(())
    z_i = F.normalize(anchor_z.view(1, -1), dim=-1)
    z_b = F.normalize(bank_z, dim=-1)
    device = z_i.device
    y_i = int(_as_1d_long(anchor_y, device)[0].item())
    s_i = int(_as_1d_long(anchor_subtype, device)[0].item())
    if s_i < 0:
        return z_i.new_zeros(())

    y_b = bank_y.to(device).long().view(-1)
    s_b = bank_subtype.to(device).long().view(-1)
    same = s_b == s_i
    if int(same.sum().item()) < 1:
        return z_i.new_zeros(())

    z_same = z_b[same]
    y_same = y_b[same]
    pos_mask = y_same == y_i
    if not bool(pos_mask.any()):
        return z_i.new_zeros(())

    logits = (z_i @ z_same.t()).view(-1) / float(temperature)
    log_prob = logits - torch.logsumexp(logits, dim=0)
    loss = -log_prob[pos_mask].mean()
    return loss


def inter_subtype_prototype_loss(anchor_z, anchor_subtype, prototypes, temperature=0.07):
    """
    L_Inter：对齐自身分型原型、排斥其他分型原型（ProtoNCE）。
    prototypes: [K, D]（建议已归一化或内部归一化）
    """
    s_i = int(_as_1d_long(anchor_subtype, anchor_z.device)[0].item())
    if s_i < 0 or prototypes is None:
        return anchor_z.new_zeros(())
    k = prototypes.size(0)
    if s_i >= k:
        return anchor_z.new_zeros(())

    z_i = F.normalize(anchor_z.view(1, -1), dim=-1)
    proto = F.normalize(prototypes, dim=-1)
    logits = (z_i @ proto.t()).view(-1) / float(temperature)  # [K]
    # 自身分型为唯一正类
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([s_i], device=z_i.device))
    return loss


class TripleLevelLoss(nn.Module):
    """
    L_Total = L_CE + α L_Global + β L_Intra + γ L_Inter
    """

    def __init__(
        self,
        temperature=0.07,
        alpha=0.5,
        beta=0.5,
        gamma=0.5,
        bank_size=256,
        proj_dim=128,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.ce = nn.CrossEntropyLoss()
        self.bank = FeatureMemoryBank(bank_size, proj_dim)

    def set_weights(self, alpha=None, beta=None, gamma=None):
        if alpha is not None:
            self.alpha = float(alpha)
        if beta is not None:
            self.beta = float(beta)
        if gamma is not None:
            self.gamma = float(gamma)

    def forward(
        self,
        logits,
        label,
        z=None,
        subtype_id=None,
        prototypes=None,
        update_bank=True,
        use_global=True,
        use_intra=True,
        use_inter=True,
    ):
        """
        Returns:
          total_loss, loss_dict
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        label_t = _as_1d_long(label, logits.device)
        loss_ce = self.ce(logits, label_t)

        loss_g = logits.new_zeros(())
        loss_intra = logits.new_zeros(())
        loss_inter = logits.new_zeros(())

        if z is not None:
            bank_z, bank_y, bank_s = self.bank.get()
            if use_global and self.alpha != 0:
                loss_g = supervised_contrastive_loss(
                    z, label_t, bank_z, bank_y, self.temperature
                )
            if use_intra and self.beta != 0 and subtype_id is not None:
                loss_intra = intra_subtype_contrastive_loss(
                    z, label_t, subtype_id, bank_z, bank_y, bank_s, self.temperature
                )
            if use_inter and self.gamma != 0 and subtype_id is not None:
                loss_inter = inter_subtype_prototype_loss(
                    z, subtype_id, prototypes, self.temperature
                )
            if update_bank:
                self.bank.update(z, label_t, subtype_id if subtype_id is not None else -1)

        total = (
            loss_ce
            + self.alpha * loss_g
            + self.beta * loss_intra
            + self.gamma * loss_inter
        )
        details = {
            "loss": float(total.detach().item()),
            "loss_ce": float(loss_ce.detach().item()),
            "loss_global": float(loss_g.detach().item()),
            "loss_intra": float(loss_intra.detach().item()),
            "loss_inter": float(loss_inter.detach().item()),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }
        return total, details
