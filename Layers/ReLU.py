import numpy as np


class ReLU:

    def __init__(self):
        self.input_tensor = None
        self.error_tensor = None
        self.gradient_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_tensor = np.maximum(np.zeros_like(input_tensor), input_tensor)
        return input_tensor
        # produce input_tensor for next layer

    def backward(self, error_tensor):
        output_tensor = error_tensor.copy()
        output_tensor[self.input_tensor <= 0] = 0
        return output_tensor
    # error_tensor = error_tensor * ReLU_derivation to Z
    # ReLU_derivation = 1 or 0
    # if the input tensor element is equal to 0, then the corresponding error tensor is 0
    # error_tensor = error_tensor[self.input_tensor == 0] = 0
    # return error_tensor
    # return a copy



