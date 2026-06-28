#!/usr/bin/env python3

from __future__ import annotations

import torch

from .constant_mean import ConstantMean


class UncertainConstantMean(ConstantMean):
    r"""
    Learned constant prior mean for (mean, variance) input pairs.

    Inputs to this module are 2 column tensors where column 0 is the input's
    mean and column 1 is its variance, the same convention used by
    :class:`gpytorch.kernels.UncertainKernel`. Only the mean column is read here; the
    variance column is read by the paired kernel module instead.

    This subclasses :class:`gpytorch.means.ConstantMean` directly and overrides only
    :meth:`forward`, so it inherits the parent's ``constant_prior``, ``constant_constraint``
    and ``batch_shape`` handling unchanged. See :class:`gpytorch.means.ConstantMean` for
    the full parameter docs.
    """

    def forward(self, input):
        mean_col = input[..., 0]
        constant = self.constant.unsqueeze(-1)  # *batch_shape x 1
        return constant.expand(torch.broadcast_shapes(constant.shape, mean_col.shape))
