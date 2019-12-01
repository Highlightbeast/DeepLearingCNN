import numpy as np
import math
import random


class Sgd:

    def __init__(self, learning_rate):
        self.learn_rate = learning_rate
        # self.weight_tensor = weight_tensor
        # self.gradient_tensor = gradient_tensor

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.learn_rate * gradient_tensor
        return weight_tensor


class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):
        self.learn_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.v * self.momentum_rate - self.learn_rate * gradient_tensor
        weight_tensor = weight_tensor + self.v
        return weight_tensor


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learn_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.t = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        epsilon = 1e-8
        # while 1:
        self.t += 1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.power(gradient_tensor, 2)

        v_hat = self.v / (1 - np.power(self.mu, self.t))
        r_hat = self.r / (1 - np.power(self.rho, self.t))
        # previous_weight = weight_tensor
        weight_tensor = weight_tensor - self.learn_rate * np.divide((v_hat + epsilon), (np.sqrt(r_hat) + epsilon))
            # if weight_tensor == previous_weight:
            #     break
        print(self.t)
        return weight_tensor



