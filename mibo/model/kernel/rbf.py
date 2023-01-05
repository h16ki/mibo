from .kernel import Kernel

import numpy as np

class RBFKernel(Kernel):
    r"""
    A covariance matrix based on the RBF Kernel:

    \begin{align}
        K(\bm{x}_1, \bm{x}_2) = \exp \Big(
            -\frac{1}{2} (\bm{x}_1 - \bm{x}_2)^T \Sigma^{-1} (\bm{x}_1 - \bm{x}_2)
        \Big)
    \end{align}

    For any set of data $\{\bm{x}^{(1)}, \cdots, \bm{x}^{(d)}$ as

    \begin{align}
        \bm{x}^{(a)} =
        \begin{pmatrix}
            \begin{pmatrix}
                x_1^{(1)} \\ \vdots \\ x_m^{(1)}
            \end{pmatrix}
            \cdots
            \begin{pmatrix}
                x_1^{(d)} \\ \vdots \\ x_m^{(d)}
            \end{pmatrix}
        \end{pmatrix}
    \end{align}

    \begin{align}
        K_{ab} = K(\bm{x}_1^{(a)}, \bm{x}_2^{(b)})
    \end{align}
    """

    def forward(self, x1, x2, **params):
        (dim, a), (_, b) = (x1.shape, x2.shape)

        kernel_shape = (a, b)
        kernel_components = [0] * (a * b)
        for i in range(a):
            for j in range(b):
                z = x1[:,i] - x2[:,j]
                kernel_components[i+j] = np.exp(
                    -0.5 * np.dot(z, z)
                )

        return np.reshape(kernel_components, newshape=kernel_shape)
