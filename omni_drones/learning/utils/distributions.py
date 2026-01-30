"""
Distribution utilities for MPC-MAPPO integration.
Based on reference implementation from MAPPO-MPC.
"""

import torch
import torch.nn as nn


class FixedNormal(torch.distributions.Normal):
    """Modify standard PyTorch Normal."""

    def log_probs(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean
