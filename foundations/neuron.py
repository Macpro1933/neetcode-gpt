import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        pre_activation = np.dot(x,w)+b
        if activation == 'sigmoid':
            ans = 1/(np.exp(-pre_activation)+1)
        elif activation == 'relu':
            ans = np.maximum(0,pre_activation)

        return np.round(ans,5)    