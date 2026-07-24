import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        x = np.array(x)  
        W1 = np.array(W1)  
        W2 = np.array(W2)  
        b1 = np.array(b1) 
        b2 = np.array(b2)
        y_true = np.array(y_true)

        def ReLU(z):
            return np.maximum(0,z)
        
        z1 = ReLU(np.dot(x,W1.T) + b1)
        z2 = np.dot(z1,W2.T) + b2

        error = 2 * (z2 - y_true) / len(y_true) 
        loss = np.round(np.mean((z2 - y_true) ** 2), 4)

        dz2 = 2 * (z2 - y_true) / len(y_true)
        dW2 = np.round(np.outer(dz2, z1),4)
        db2 = np.round(dz2,4)

        da1 = np.dot(dz2, W2)
        dz1 = da1 * (z1 > 0)

        dW1 = np.round(np.outer(dz1, x),4)
        db1 = np.round(dz1,4)

        return {
            "loss" : loss,
            "dW1" : dW1.tolist(),
            "db1" : db1.tolist(),
            "dW2" : dW2.tolist(),
            "db2" : db2.tolist()
        }