#!/usr/bin/env python3

from __future__ import annotations

import torch

from .kernel import Kernel


class UncertainKernel(Kernel):
    r"""
    RBF style kernel for one dimensional inputs whose locations are only
    known up to a Gaussian uncertainty.

    Implements the covariance function from Dallaire, Besse and Chaib draa,
    "Learning Gaussian Process Models from Uncertain Data" (ICONIP 2009),
    which marginalizes the input noise out of a standard RBF kernel in
    closed form.

    Each input point is encoded as a 2 column tensor, not a single value:

    .. math::
        x_{..., 0} = \text{mean of the input location} \\
        x_{..., 1} = \text{variance of the input location (0 for an exactly known point)}

    For two points :math:`x_i = (\mu_i, s_i^2)` and :math:`x_j = (\mu_j, s_j^2)`, with
    lengthscale :math:`l`, the kernel is:

    .. math::
        \begin{equation*}
            \text{norm}_{ij} = l^2 + s_i^2 + s_j^2 \\
            k(x_i, x_j) = \sqrt{l^2 / \text{norm}_{ij}} \exp(-0.5 (\mu_i - \mu_j)^2 / \text{norm}_{ij})
        \end{equation*}

    When :math:`s_i = s_j = 0` this reduces exactly to a standard RBF kernel. Larger
    input variance on either point widens :math:`\text{norm}_{ij}`, which both flattens the
    exponential term and shrinks the sqrt prefactor, so uncertain points
    contribute weaker covariance than precisely known ones.

    Wrap this kernel in :class:`gpytorch.kernels.ScaleKernel` to add a learned output
    scale (signal variance).
    """

    has_lengthscale = True

    @staticmethod
    def _split(x):
        """Split a (..., N, 2) tensor of (mean, variance) pairs into its two columns."""
        mean = x[..., 0:1]
        var = x[..., 1:2]
        return mean, var

    def forward(self, x1, x2, diag=False, **params):
        x1_mean, x1_var = self._split(x1)
        x2_mean, x2_var = self._split(x2)

        l2 = self.lengthscale.pow(2)

        # Squared distance between means, shape (..., N, M), or (..., N) if diag.
        sq_dist = self.covar_dist(x1_mean, x2_mean, diag=diag, square_dist=True)

        if diag:
            # x1_var and x2_var are already aligned point for point.
            norm = (l2 + x1_var + x2_var).squeeze(-1)
        else:
            # Pairwise sum: norm[..., i, j] = l^2 + var1[i] + var2[j]
            norm = l2 + x1_var + x2_var.transpose(-1, -2)

        return (l2 / norm).sqrt() * torch.exp(-0.5 * sq_dist / norm)
