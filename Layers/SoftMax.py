import numpy as np


class SoftMax:

    def __init__(self):
        self.prediction = None

    def forward(self, input_tensor):
        x_max = np.max(input_tensor)
        shifted_input_tensor = input_tensor - x_max
        softmax = np.exp(shifted_input_tensor)
        sum_softmax = np.sum(softmax, axis=1, keepdims=True)
        self.prediction = np.divide(softmax, sum_softmax)
        # softmax = exp(x_k) / sum(exp(x_j)) since input_tensor is given with size (batch_size * j+1)
        # we want to get the sum for each batch, axis = 1, sum over rows.
        return np.copy(self.prediction)

    def backward(self, error_tensor):
        error_over_batched = np.sum(self.prediction * error_tensor, axis=1, keepdims=True)
        error_tensor = self.prediction * (error_tensor - error_over_batched)
        return error_tensor






