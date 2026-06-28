#!/usr/bin/env python3

from __future__ import annotations

import torch

from .mean import Mean


class UncertainZeroMean(Mean):
    r"""
    Zero prior mean for (mean, variance) input pairs.

    Inputs to this module are 2 column tensors where column 0 is the input's
    mean and column 1 is its variance, the same convention used by
    :class:`gpytorch.kernels.UncertainKernel`. Only the mean column is read here; the
    variance column is read by the paired kernel module instead.
    """

    def __init__(self, batch_shape: torch.Size = torch.Size(), **kwargs):
        super().__init__()
        self.batch_shape = batch_shape

    def forward(self, input):
        mean_col = input[..., 0]
        # Trailing size 1 dim lets this broadcast against mean_col's last
        # (point count) dimension, whatever it is.
        zeros = torch.zeros(*self.batch_shape, 1, dtype=mean_col.dtype, device=mean_col.device)
        return zeros.expand(torch.broadcast_shapes(zeros.shape, mean_col.shape))
