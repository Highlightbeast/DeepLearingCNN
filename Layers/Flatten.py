import numpy as np


class Flatten:
    def __init__(self):
        self.input_tensor = None
    #     input tensor size = [batch_size, channel, y, x]

    # which reshapes the input tensor to [batch_size, c*y*x] before fc layer

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_tensor = np.reshape(input_tensor, (input_tensor.shape[0], -1))
        return input_tensor

    #  which reshapes and returns the error tensor to [batch_size, channel, y, x]
    def backward(self, error_tensor):
        # return 1d error to 4d
        error_tensor = np.reshape(error_tensor, self.input_tensor.shape)
        return error_tensor

