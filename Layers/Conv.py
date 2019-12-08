import numpy as np
import math
import scipy.signal as sgl
import copy

class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        if len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        self.stride_shape = stride_shape
        # define the size of kernel (c, m, n) c is channel, m*n is convolution range
        # if input kernel is (c, m), generalize it into (c, m, 1)
        if len(convolution_shape) == 2:
            convolution_shape = (convolution_shape[0], convolution_shape[1], 1)
        self.X = None
        self.Y = None
        self.num_y = None
        self.num_x = None
        self.input_tensor = None
        self.num_kernels = num_kernels
        self.convolution_shape = convolution_shape
        self.weights = np.random.random_sample(np.concatenate(([self.num_kernels], self.convolution_shape)))
        self.bias = np.random.rand(self.num_kernels, 1)
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        self._optimizer_weights = None
        self._optimizer_bias = None

    def forward(self, input_tensor):
        # input_tensor shape (B, C, Y, X). C is the num_channels, B is the num_batches
        # padded_input_tensor shape (B, C, X + 2 * floor(M/2) )
        # weight shape (H, C, M, N). H is the num_kernels
        # output shape (B, H, Y', X')

        # generalization input_tensor

        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=3)

        self.input_tensor = input_tensor

        self.Y = input_tensor.shape[2]
        self.X = input_tensor.shape[3]

        # output_tensor (for sub-sampling)
        self.num_y = math.ceil(self.Y / self.stride_shape[0])
        self.num_x = math.ceil(self.X / self.stride_shape[1])

        output_tensor = np.zeros((self.input_tensor.shape[0], self.num_kernels, self.Y, self.X))
        output_tensor_sub = np.zeros((self.input_tensor.shape[0], self.num_kernels, self.num_y, self.num_x))

        # do correlation, stride-shape is 1
        for b in range(self.input_tensor.shape[0]):
            for h in range(self.num_kernels):
                # third loop over channel
                for c in range(self.input_tensor.shape[1]):
                    output_tensor[b, h, :, :] += sgl.correlate(input_tensor[b, c, :, :], self.weights[h, c, :, :],
                                                               mode='same')

                # add bias
                output_tensor[b, h, :, :] = output_tensor[b, h, :, :] + self.bias[h]

                # do sub-sampling for stride-shape isn't 1
                output_tensor_sub[b, h, :, :] = output_tensor[b, h, ::self.stride_shape[0], ::self.stride_shape[1]]

        # if 1D case, change output from (B, H, X', 1) to (B, H, X')
        if output_tensor_sub.shape[3] == 1:
            output_tensor_sub = np.reshape(output_tensor_sub,
                                           (self.input_tensor.shape[0], self.num_kernels, output_tensor_sub.shape[2]))
        return output_tensor_sub
    #
    def backward(self, error_tensor):

        if len(error_tensor.shape) == 3:
            error_tensor = np.expand_dims(error_tensor, axis=3)

        input_tensor = copy.deepcopy(self.input_tensor)
        temp = np.zeros((self.input_tensor.shape[0], self.num_kernels, self.Y, self.X))
        # up-sampling error_tensor, if stride size is 2d
        temp[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
        error_tensor = temp

        # error_tensor_sub = temp[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
        # output tensor in backward is the error tensor of last layer,
        # which corresponds to the input_tensor of the last layer
        # output_tensor [B, H, Y, X]
        output_tensor = np.zeros((self.input_tensor.shape[0], self.input_tensor.shape[1], self.Y, self.X))
        # deepcopy kernels to avoid unnecessary modifications
        backward_kernel = copy.deepcopy(self.weights)

        # update error tensor for last layer = error tensor * backward_kernel)
        # error tensor[B, H, Y, X] 和backward kernel [H, C, Y, X]卷积的时候mode用same，保证输出是[B, C, Y, X]
        # forward calculate output tensor with correlation, backward with convolution
        # first loop over batch, batch_size = self.input_tensor.shape[0]
        # backward_kernel = np.reshape(backward_kernel, (self.input_tensor.shape[0], self.num_kernels, -1, -1))
        for b in range(self.input_tensor.shape[0]):
            # second loop over channel
            for c in range(self.input_tensor.shape[1]):
                # third loop over kernel
                for h in range(self.num_kernels):
                    output_tensor[b, c, :, :] += sgl.convolve2d(error_tensor[b, h, :, :], backward_kernel[h, c, :, :],
                                                                mode='same')

        # update gradient(weights and bias)
        # self.input_tensor[b, c, y, x] convolve error tensor[b, h, y', x']
        # weight gradient[num_kernels, c, m, n]
        # initialize gradient tensor
        self._gradient_weights = np.zeros(np.concatenate(([self.num_kernels], self.convolution_shape)))
        self._gradient_bias = np.zeros_like(self.bias)
        y_pad_size = int((self.convolution_shape[1] - 1) // 2)
        x_pad_size = int((self.convolution_shape[2] - 1) // 2)
        if (self.convolution_shape[1] % 2 == 1) and (self.convolution_shape[2] % 2 == 1):
            pad_input_tensor = np.pad(input_tensor,
                                      ((0, 0), (0, 0), (y_pad_size, y_pad_size), (x_pad_size, x_pad_size)),
                                      mode='constant', constant_values=0)
        elif (self.convolution_shape[1] % 2 == 1) and (self.convolution_shape[2] % 2 == 0):
            x_pad_size_l = x_pad_size + 1
            pad_input_tensor = np.pad(input_tensor,
                                      ((0, 0), (0, 0),
                                       (y_pad_size, y_pad_size), (x_pad_size_l, x_pad_size)),
                                      mode='constant', constant_values=0)
        elif (self.convolution_shape[1] % 2 == 0) and (self.convolution_shape[2] % 2 == 1):
            y_pad_size_u = y_pad_size + 1
            pad_input_tensor = np.pad(input_tensor,
                                      ((0, 0), (0, 0), (y_pad_size_u, y_pad_size), (x_pad_size, x_pad_size)),
                                      mode='constant', constant_values=0)
        else:
            x_pad_size_l = x_pad_size + 1
            y_pad_size_u = y_pad_size + 1
            pad_input_tensor = np.pad(input_tensor,
                                      ((0, 0), (0, 0), (y_pad_size_u, y_pad_size), (x_pad_size_l, x_pad_size)),
                                      mode='constant', constant_values=0)
        # first loop over num_kernel
        for h in range(self.num_kernels):
            # second loop over channels
            for b in range(self.input_tensor.shape[0]):
                # third loop over over batch size
                for c in range(input_tensor.shape[1]):
                    self._gradient_weights[h, c, :, :] += sgl.correlate(pad_input_tensor[b, c, :, :], error_tensor[b, h, :, :],
                                                                       mode='valid')
                self._gradient_bias[h] += np.sum(error_tensor[b, h, :, :])
        # self.gradient_bias = np.sum(np.sum(np.sum(error_tensor, axis=3), axis=2), axis=0)
        # 1D case
        if output_tensor.shape[3] == 1:
            output_tensor = np.reshape(output_tensor, (self.input_tensor.shape[0], self.input_tensor.shape[1], self.Y))

        # update weights and bias according to some optimizer gradient algorithm
        if self._optimizer is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)
        return output_tensor

    def initialize(self, weights_initializer, bias_initializer):
        # fan_in = input_channel * kernel_height * kernel_weight
        # fan_out = output_channel * kernel_height * kernel_weight
        #fan_in = self.input_tensor.shape[1] * self.convolution_shape[1] * self.convolution_shape[2]
        #fan_out = self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2]
        fan_in = self.weights.shape[1] * np.prod(self.convolution_shape[1:])
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    def get_gradient_weights(self):
        return self._gradient_weights

    def set_gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    gradient_weights = property(get_gradient_weights, set_gradient_weights)

    def get_gradient_bias(self):
        return self._gradient_bias

    def set_bias_weights(self, gradient_bias):
        self._gradient_bias = gradient_bias
    gradient_bias = property(get_gradient_bias, set_bias_weights)

    # property optimizer

    def get_optimizier(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_weights = copy.deepcopy(self._optimizer)
        self._optimizer_bias = copy.deepcopy(self._optimizer)

    optimizer = property(get_optimizier, set_optimizer)

