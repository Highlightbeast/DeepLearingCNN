import numpy as np
import copy
from scipy import signal
class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.input_tensor = None
        # self.input_tensor_shape = None
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        # Initialize the parameters of this layer uniformly random in the range[0, 1).
        # self.weights = np.random.rand(self.num_kernels, self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2])
        self.weights = np.random.random(np.concatenate(([self.num_kernels], self.convolution_shape)))
        self.bias = np.random.rand(self.num_kernels)
        self.step_x = None
        self.step_y = None
        self.gradient_weights = None
        self.gradient_bias = None



    def forward(self, input_tensor):
        # input_tensor shape
        # 1D:[b,c,y];
        # 2D:[b,c,y,x]
        # output_tensor shape: [b,H,y,x] y,x are different from x,y in input_tensor
        # H is num_kernels
        self.input_tensor = input_tensor
        # if self.convolution_shape[1] % 2 == 1:
        #     y_pad_size = int((self.convolution_shape[1] - 1) // 2)
        # else:
        #     y_pad_size = int((self.convolution_shape[1]))
        y_pad_size = int((self.convolution_shape[1] - 1) // 2)
        if len(self.convolution_shape) == 3:

            x_pad_size = int((self.convolution_shape[2] - 1)//2)
            # if (y_pad_size % 2 == 1) and (x_pad_size % 2 == 1):
            pad_input = np.pad(input_tensor, ((0, 0), (0, 0), (y_pad_size, y_pad_size), (x_pad_size + 1, x_pad_size)),
                               mode='constant', constant_values=0)
            output_tensor = np.zeros((self.input_tensor.shape[0], self.num_kernels,
                                      ((pad_input.shape[2] - self.convolution_shape[1])//self.stride_shape[0] + 1),
                                      ((pad_input.shape[3] - self.convolution_shape[2])//self.stride_shape[1] + 1)))

            # first loop over all the batches
            for b in range(self.input_tensor.shape[0]):
                # second loop over all kernels, which is also the depth of output_tensor
                batch_temp = np.zeros((self.num_kernels, input_tensor.shape[2], input_tensor.shape[3]))
                for h in range(self.input_tensor.shape[1]):
                    # 3D convolution or correlation
                    # batch_1 output_tensor [h, y, x]
                    # kernel_1 weights tensor [h, convolution_shape[1], convolution_shape[2]]
                    temp = signal.correlate(pad_input[b], self.weights[h], 'valid')
                    bias_tensor = np.ones_like(temp) * self.bias[h]
                    temp += bias_tensor
                    # print(temp.shape)
                    # each kernel has one bias, bias tensor should corresponds to the output tensor for each kernel
                    # bias_tensor = np.ones_like(temp) * self.bias[h]
                    # output_tensor[b, h, :, :] = output_tensor[b, h, :, :] + bias_tensor
                    # down_sampling with stride size
                    batch_temp[h] = temp
                batch_temp = batch_temp[:, ::self.stride_shape[0], :: self.stride_shape[1]]
                output_tensor[b] = batch_temp
            self.input_tensor = input_tensor
                # input_tensor[b] = input_tensor[:, ::self.stride_shape[0], :: self.stride_shape[1]]
                # output_tensor[b] = input_tensor[b]
        else:
            pad_input = np.pad(input_tensor, ((0, 0), (0, 0), (y_pad_size, y_pad_size)), mode='constant',
                               constant_values=0)
            output_tensor = np.zeros((self.input_tensor.shape[0], self.num_kernels,
                                      ((pad_input.shape[2] - self.convolution_shape[1])//self.stride_shape[0] + 1)))

            # first loop over all the batches
            for b in range(self.input_tensor.shape[0]):
                # second loop over all kernels, which is also the depth of output_tensor
                batch_temp = np.zeros((self.num_kernels, input_tensor.shape[2]))
                for h in range(self.input_tensor.shape[1]):
                    # 2D convolution or correlation
                    # batch_1 output_tensor [h, y]
                    # kernel_1 weights tensor [h, convolution_shape[1]]
                    temp = signal.correlate(pad_input[b], self.weights[h], 'valid')
                    bias_tensor = np.ones_like(temp) * self.bias[h]
                    temp += bias_tensor
                    # down_sampling with stride size
                    batch_temp[h] = temp
                batch_temp = batch_temp[:, ::self.stride_shape[0]]
                output_tensor[b] = batch_temp
            self.input_tensor = input_tensor
        return output_tensor

                    # third loop over all channels(depth of input)
                    # 有点麻烦，开始
                    # for c in range(self.input_tensor.shape[1]):
                    #     # 1D case
                    #     if self.input_tensor.ndim == 4:
                    #         output_tensor[b, h, :, :] += signal.correlate2d(input_tensor[b, c, :, :], self.weights[h, c, :, :], mode='same')
                    #     # 2D case
                    #     else:
                    #         output_tensor[b, h, :] += signal.correlate(input_tensor[b, c, :], self.weights[h, c, :], mode='same')

                # # Bias in 1D case
                # if self.input_tensor.ndim == 4:
                #     # each kernel has one bias. In vector self.bias, there are num_kernels elements
                #     bias_tensor = np.ones_like(output_tensor[b, h, :, :]) * self.bias[h]
                #     output_tensor[b, h, :, :] += bias_tensor
                # # Bias in 2D case
                # else:
                #     bias_tensor = np.ones_like(output_tensor[b, h, :]) * self.bias[h]
                #     output_tensor[b, h, :] += bias_tensor
                # 有点麻烦, 结束

        # if len(self.stride_shape) == 2:
        #     # down-sampling index of y and axis
        #     self.step_y = self.stride_shape[1] * np.arange(np.ceil(output_tensor.shape[2] / self.stride_shape[1])).astype(int)
        #     self.step_x = self.stride_shape[0] * np.arange(np.ceil(output_tensor.shape[3] / self.stride_shape[0])).astype(int)
        #     output_tensor = output_tensor[:, :, self.step_y, :]
        #     output_tensor = output_tensor[:, :, :, self.step_x]
        # else:
        #     self.step_y = self.stride_shape[0] * np.arange(np.ceil(output_tensor.shape[2] / self.stride_shape[0])).astype(int)
        #     output_tensor = output_tensor[:, :, self.step_y]
        #
        #
        # return output_tensor

    # def backward(self, error_tensor):
    #     temp = np.zeros(np.concatenate((error_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2:])))
    #




    @property
    def get_gradient_weights(self):
        return self.gradient_weights

    def set_gradient_weights(self, gradient_weights):
        self.gradient_weights = gradient_weights

    def get_gradient_bias(self):
        return self.gradient_bias

    def set_gradient_weights(self, gradient_bias):
        self.gradient_bias = gradient_bias

