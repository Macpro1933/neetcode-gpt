import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        y_pred += 1e-7
        loss = np.sum(y_true*np.log(y_pred)+ (1-y_true)*np.log(1-y_pred))
        return np.round(-loss/len(y_true),4) 
        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n_classes = np.shape(y_true)[1]
        n_samples = np.shape(y_true)[0]
        y_pred+=1e-7
        loss = 0

        for i in range(n_samples):
            for j in range(n_classes):
                loss+=y_true[i][j]*np.log(y_pred[i][j])
        return np.round(-loss/n_samples,4)