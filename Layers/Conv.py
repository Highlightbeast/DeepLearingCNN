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
        # weight shape, each kernel has one weight
        # each weight shape corresponds to convolution shape
        # total weight shape corresponds to [num_kernel, convolution shape]
        # 3D [num_kernels, self.convolution_shape[1], self.convolution_shape[2]]
        # 2D [num_kernels, self.convolution_shape[1]]
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
        # output_tensor shape: [b,h,y,x] y,x are different from x,y in input_tensor
        # h is num_kernels
        self.input_tensor = input_tensor

        if len(self.convolution_shape) == 3:
            y_pad_size = int((self.convolution_shape[1] - 1) // 2)
            x_pad_size = int((self.convolution_shape[2] - 1) // 2)
            # different convolution_shape leads to different padding size
            # odd--> both side is padded with (convolution_size - 1) / 2
            # even--> one side (convolution_size - 1) / 2; other side (convolution_size - 1) / 2 + 1
            # 比如convolution_shape(2，3，4)
            # y方向3，正常padding，（3-1）// 2 = 1上下都来1行1列
            # x方向讨厌是个偶数4，（4-1）// 2 = 1，要是左右都padding 1行1列不够的，只能左边2个右边1个
            # 这样correlate或者convolution之后的output size 才会和input size保持不变
            if (self.convolution_shape[1] % 2 == 1) and (self.convolution_shape[2] % 2 == 1):
                pad_input = np.pad(input_tensor, ((0, 0), (0, 0),
                                                  (y_pad_size, y_pad_size),
                                                  (x_pad_size, x_pad_size)),
                                   mode='constant', constant_values=0)
            elif (self.convolution_shape[1] % 2 == 1) and (self.convolution_shape[2] % 2 == 0):
                x_pad_size_l = x_pad_size + 1
                pad_input = np.pad(input_tensor, ((0, 0), (0, 0),
                                                  (y_pad_size, y_pad_size),
                                                  (x_pad_size_l, x_pad_size)),
                                   mode='constant', constant_values=0)
            elif (self.convolution_shape[1] % 2 == 0) and (self.convolution_shape[2] % 2 == 1):
                y_pad_size_u = y_pad_size + 1
                pad_input = np.pad(input_tensor, ((0, 0), (0, 0),
                                                  (y_pad_size_u, y_pad_size),
                                                  (x_pad_size, x_pad_size)),
                                   mode='constant', constant_values=0)
            else:
                x_pad_size_l = x_pad_size + 1
                y_pad_size_u = y_pad_size + 1
                pad_input = np.pad(input_tensor, ((0, 0), (0, 0),
                                                  (y_pad_size_u, y_pad_size),
                                                  (x_pad_size_l, x_pad_size)),
                                   mode='constant', constant_values=0)
            # 计算output_tensor shape的大小, 如果stride = 1， output size 和input size一样，但是如果有stride，就要这样重新计算
            output_tensor = np.zeros((self.input_tensor.shape[0], self.num_kernels,
                                      ((pad_input.shape[2] - self.convolution_shape[1])//self.stride_shape[0] + 1),
                                      ((pad_input.shape[3] - self.convolution_shape[2])//self.stride_shape[1] + 1)))

            # first loop over all the batches
            for b in range(self.input_tensor.shape[0]):
                # second loop over all kernels, which is also the depth of output_tensor
                batch_temp = np.zeros((self.num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3]))
                for h in range(self.num_kernels):
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
            # self.input_tensor = input_tensor

        else:
            y_pad_size = int((self.convolution_shape[1] - 1) // 2)
            if self.convolution_shape[1] % 2 == 1:
                pad_input = np.pad(input_tensor, ((0, 0), (0, 0),
                                                  (y_pad_size, y_pad_size)),
                                   mode='constant', constant_values=0)
            else:
                y_pad_size_u = y_pad_size + 1
                pad_input = np.pad(input_tensor, ((0, 0), (0, 0),
                                                  (y_pad_size_u, y_pad_size)),
                                   mode='constant', constant_values=0)

            output_tensor = np.zeros((self.input_tensor.shape[0], self.num_kernels,
                                      ((pad_input.shape[2] - self.convolution_shape[1])//self.stride_shape[0] + 1)))

            # first loop over all the batches
            for b in range(self.input_tensor.shape[0]):
                # second loop over all kernels, which is also the depth of output_tensor
                batch_temp = np.zeros((self.num_kernels, self.input_tensor.shape[2]))
                for h in range(self.num_kernels):
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
            # self.input_tensor = input_tensor
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





    @property
    def get_gradient_weights(self):
        return self.gradient_weights

    def set_gradient_weights(self, gradient_weights):
        self.gradient_weights = gradient_weights

    def get_gradient_bias(self):
        return self.gradient_bias

    def set_gradient_weights(self, gradient_bias):
        self.gradient_bias = gradient_bias

