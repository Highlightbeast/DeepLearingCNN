import random
import numpy as np
import copy
class FullyConnected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None
        self.error_tensor = None
        self.weights = np.random.rand(self.output_size, self.input_size + 1)
        self.input_tensor = None
        self.output_tensor = None
        self.gradient_tensor = None

    # sets and returns (getter) the protected member optimizer
    # getter
    @property
    def get_optimizer(self):
        return self._optimizer
    # setter

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
    optimizer = property(get_optimizer, set_optimizer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        ones = np.ones([batch_size, 1])
        # ones = np.ones([input_tensor.shape[0], 1])
        # last column is 'one' vector, length of this vector is batch_size, since each batch has a bias weight
        bias_input_tensor = np.concatenate((input_tensor, ones), axis=1)
        # bias_input_tensor shape = b x (j+1)
        input_tensor = np.dot(bias_input_tensor, self.weights.T)
        # self.weight shape = k x j
        return input_tensor
        # produce input_tensor for next layer

    def backward(self, error_tensor):

        batch_size = self.input_tensor.shape[0]
        ones = np.ones([batch_size, 1])
        bias_input_tensor = np.concatenate((self.input_tensor, ones), axis=1)
        self.gradient_tensor = np.dot(bias_input_tensor.T, error_tensor).T
        update_error_tensor = np.dot(error_tensor, self.weights)
        error_tensor = np.delete(update_error_tensor, -1, axis=1)
        # do not perform an update if the optimizer is unset
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_tensor)

        return error_tensor
        # error_tensor.shape = batch_size * k, k is output_size
        # transpose of the error_tensor that is given in the slides
        #  calculate the gradient of weights
        # produce the bias_input_tensor. given input_tensor doesn't including bias.

        # print(bias_input_tensor.shape)
        # bias_input_tensor.shape = batch_size * (j + 1), j is input_size

        # w' = w'-learning_rate * bias_input_tensor.T * error_tensor
        # all the formula in the paper: self.gradient_weight = bias_input_tensor.T * error_tensor
        #  (j + 1) * batch_size multiple with batch_size * k, therefore, self.gradient_weight.shape = (j + 1) * k
        # but all the things in this program are given in the transpose type
        # if we do not transpose the result of (bias_input_tensor.T * error_tensor)
        # the dimension of this result will not correspond to the weights.
        # the dimension of weights is always! k * (j + 1)
        # print(self.gradient_weight)
        # print(self.gradient_weight.shape)
        # update error tensor for last layer

        # update_error_tensor.shape = batch_size * (j + 1)
        # remove the last column of this new error_tensor

        # error_tensor.shape = batch_size * j

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.input_size
        fan_out = self.output_size
        # bias has the same size with output
        bias = np.zeros(self.output_size)
        weights = weights_initializer.initialize(self.weights[:, :-1].shape, fan_in, fan_out)
        bias = bias_initializer.initialize(bias.shape, fan_in, fan_out)
        self.weights = np.concatenate((weights, np.expand_dims(bias, axis=1)), axis=1)

    def set_optimizer(self, optimizer):
        self.optimizer = copy.deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self.gradient_tensor
    #

    #
    #
    #
    #
    # def set_optimizer(self, optimizer):


