from __future__ import annotations
from typing import TypeAlias

import numpy as np
from model.mean import Mean
from model.kernel import Kernel


NDArray: TypeAlias = np.typing.NDArray
MeanMod: TypeAlias = Mean
KernMod: TypeAlias = Kernel

class GaussianProcess():
    r"""
    Args:
        train_in: A (n, d) shape array of training features.
        train_out: A (n, m) shape array training observations.
        mean: The module computing the mean function.
            If none, use a 'ConstantMean'.
        kernel: The module computing the kernel function.
            If none, use a 'RBFKernel'.

    Example:
    >>> train_in = ...; train_out = ...
    >>> model = GaussianProcess(train_in, train_out, mean=..., kernel=...)
    >>> test_x = ...
    >>> loc, interval = model.likelihood(test_x, sigma_level=1)
    >>> # UBC = loc + interval, LBC = loc - interval
    >>> model.sample(test_x, n=50)  # Box-muller's method
    """
    def __init__(self,
        train_in: NDArray,
        train_out: NDArray,
        mean: MeanMod = None,
        kernel: KernMod = None) -> None:

        self.mean = mean
        self.kernel = kernel

    def sample(self):
        pass
