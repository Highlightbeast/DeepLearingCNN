import numpy as np


class Constant:
    def __init__(self):
        self.weight_initialize = 0.1

    def initialize(self, weights_shape, fan_in, fan_out):
        initialized_tensor = self.weight_initialize * np.ones(weights_shape)
        return initialized_tensor


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        initialized_tensor = np.random.rand(fan_in, fan_out)
        return initialized_tensor


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2.0 / (fan_out + fan_in))
        initialized_tensor = np.random.normal(0, sigma, weights_shape)
        return initialized_tensor


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2.0 / fan_in)
        initialized_tensor = np.random.normal(0, sigma, weights_shape)
        return initialized_tensor