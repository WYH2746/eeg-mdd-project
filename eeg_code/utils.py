# eeg_code/utils.py
"""
Utility helpers for EEG-MDD project
----------------------------------
* set_seed  —— 固化随机种子
* focal_loss —— 处理类别不平衡的 Focal Loss
"""

import random, os
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For extra determinism (可选)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Compute Focal Loss (binary/multi-class).

    Args
    ----
    logits : (B, num_classes)  raw outputs from network
    targets: (B,)  ground-truth class indices
    alpha  : balance factor for rare class
    gamma  : focusing parameter

    Returns
    -------
    torch.Tensor  scalar loss
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)          # probability of true class
    loss = alpha * (1.0 - pt) ** gamma * ce
    return loss.mean()
