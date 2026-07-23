import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.round(np.linalg.matmul(X,weights),5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        sqr = (model_prediction-ground_truth)**2
        sqr_sum = np.sum(sqr)
        final = (1/np.shape(ground_truth)[0])*sqr_sum
        return np.round(final,5)
