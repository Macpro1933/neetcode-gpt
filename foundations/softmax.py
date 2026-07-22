import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        value = np.exp(z-max(z))/np.sum(np.exp(z-max(z)))
        return np.round(value,4)
