# loss.py
# Hierarchical loss for Option A1 (two heads):
#   L = lambda_parent * CE(parent_logits, parent_labels)
#     + lambda_leaf   * CE(leaf_logits,   leaf_labels)
#
# Expected inputs:
# - parent_logits: [B, P] (P=5)
# - leaf_logits:   [B, L] (L=23)
# - parent_labels: [B] int64
# - leaf_labels:   [B] int64


from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalLoss(nn.Module):
    """
    Hierarchical loss module for two-head classification.
    """

    def __init__(
        self,
        lambda_parent: float = 1.0,
        lambda_leaf: float = 1.0,
    ):
        super().__init__()
        self.lambda_parent = lambda_parent
        self.lambda_leaf = lambda_leaf

    def forward(
        self,
        parent_logits: torch.Tensor,
        leaf_logits: torch.Tensor,
        parent_labels: torch.Tensor,
        leaf_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns scalar total loss.
        """
        # ----------------------------
        # Cross-entropy losses
        # ----------------------------
        loss_parent = F.cross_entropy(
            parent_logits,
            parent_labels,
        )
        loss_leaf = F.cross_entropy(
            leaf_logits,
            leaf_labels,
        )

        total = self.lambda_parent * loss_parent + self.lambda_leaf * loss_leaf
        return total

    @torch.no_grad()
    def breakdown(
        self,
        parent_logits: torch.Tensor,
        leaf_logits: torch.Tensor,
        parent_labels: torch.Tensor,
        leaf_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Returns a Python dict of scalar floats for logging.
        Does not backprop.
        """
        loss_parent = F.cross_entropy(
            parent_logits,
            parent_labels,
        )
        loss_leaf = F.cross_entropy(
            leaf_logits,
            leaf_labels,
        )

        out = {
            "loss_parent": float(loss_parent.item()),
            "loss_leaf": float(loss_leaf.item()),
            "lambda_parent": float(self.lambda_parent),
            "lambda_leaf": float(self.lambda_leaf),
        }


        out["loss_total"] = float(
            (self.lambda_parent * loss_parent + self.lambda_leaf * loss_leaf).item()
        )
        return out