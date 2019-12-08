import numpy as np
import math


class Pooling:

    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None

        self.num_y = None
        self.num_x = None
        self.B = None
        self.C = None
        self.Y = None
        self.X = None

    def forward(self, input_tensor):

        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=3)

        self.input_tensor = input_tensor
        self.B = input_tensor.shape[0]
        self.C = input_tensor.shape[1]
        self.Y = input_tensor.shape[2]
        self.X = input_tensor.shape[3]
        self.num_y = math.floor((self.Y - self.pooling_shape[0]) / self.stride_shape[0] + 1)
        self.num_x = math.floor((self.X - self.pooling_shape[1]) / self.stride_shape[1] + 1)
        # output_tensor size[b, c, y, x]
        output_tensor = np.zeros((self.B, self.C, self.num_y, self.num_x))
        # self.position = np.zeros((self.B, self.C, self.num_y * self.num_x))
        # this list stores the position information, which batch, channel, y and x
        # 不能定义在结构函数里，因为实在forward里面需要更新的，而且backward需要用
        self.position = []
        for b in range(self.B):
            for c in range(self.C):
                for y in range(self.num_y):
                    for x in range(self.num_x):
                        mask = input_tensor[b, c,
                                 y * self.stride_shape[0]: y * self.stride_shape[0] + self.pooling_shape[0],
                                 x * self.stride_shape[1]: x * self.stride_shape[1] + self.pooling_shape[1]]
                        output_tensor[b, c, y, x] = np.max(mask)
                        max_index = np.where(mask == np.max(mask))
                        y_index = max_index[0] + self.stride_shape[0] * y
                        x_index = max_index[1] + self.stride_shape[1] * x
                        self.position.append([b, c, y_index[0], x_index[0]])
        if output_tensor.shape[3] == 1:
            output_tensor = np.reshape(output_tensor, (self.B, self.C, self.num_y))
        return output_tensor

    def backward(self, error_tensor):

        if len(error_tensor.shape) == 3:
            error_tensor = np.expand_dims(error_tensor, axis=3)
        # output_tensor [b,c,y,x]
        output_tensor = np.zeros((self.B, self.C, self.Y, self.X))
        # reshape error tensor to [不知道多少行自己算，但是我要一列]
        temp = np.reshape(error_tensor, (-1, 1))
        for i in range(len(self.position)):
            b = self.position[i][0]
            c = self.position[i][1]
            y = self.position[i][2]
            x = self.position[i][3]
            output_tensor[b, c, y, x] += temp[i]
        if output_tensor.shape[3] == 1:
            output_tensor = np.reshape(output_tensor, self.B, self.C, self.num_y)
        return output_tensor







