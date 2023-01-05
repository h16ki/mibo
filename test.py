import numpy as np
import mibo
from mibo.model.kernel.rbf import RBFKernel


if __name__ == "__main__":
    model = RBFKernel(1, 2)
    model(3)
