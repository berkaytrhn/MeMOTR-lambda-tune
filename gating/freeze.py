"""Freeze everything except the newly-added gating modules.

The new gate submodules are attached to QueryUpdater under the marker
attribute names `lt_gate` and `st_gate`, so their parameter names contain
".lt_gate." / ".st_gate.". We freeze by requires_grad; get_param_groups in
train_engine already filters by requires_grad, so the surviving params land
in the existing query_updater LR group.
"""
from typing import List

import torch.nn as nn


def freeze_all_but_gates(model: nn.Module) -> List[str]:
    """Set requires_grad=True only on gate params; freeze the rest.

    Returns the list of trainable parameter names (caller should assert it is
    non-empty and matches the experiment's expected param count).
    """
    trainable = []
    for n, p in model.named_parameters():
        parts = n.split(".")
        keep = ("lt_gate" in parts) or ("st_gate" in parts)
        p.requires_grad_(keep)
        if keep:
            trainable.append(n)
    return trainable
