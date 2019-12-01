import numpy as np
import copy


class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.forward()
        # get the initial input_tensor and the label_tensor, which is provided by data_layer.forward function
        for layer in self.layers:
            # return the input_tensor for next layer
            # in each layer we call the forward function to calculate input_tensor for next layer
            input_tensor = layer.forward(input_tensor)
        loss_entropy = self.loss_layer.forward(input_tensor, self.label_tensor)
        return loss_entropy

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        return error_tensor

    def append_trainable_layer(self, layer):
        layer.optimizer = copy.deepcopy(self.optimizer)
        layer.initialize = (copy.deepcopy(self.weights_initializer), copy.deepcopy(self.bias_initializer))
        self.layers.append(layer)

    def train(self, iterations):
        # for each iteration, call forward and backward propagation parts.
        for i in range(iterations):
            # store loss for each iteration
            loss_entropy = self.forward()
            self.loss.append(loss_entropy)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            # prediction = layer.forward(input_tensor)
            # input_tensor = np.copy(prediction)
            input_tensor = layer.forward(input_tensor)
        return input_tensor


















