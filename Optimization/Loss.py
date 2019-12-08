import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        # for every y and y.hat in the input tensor and label tensor
        # do cross_entropy formula  - y * log(y.hat), 在收到y.hat的情况下，理想值是y的概率
        # p（也就是y）表示真实标记的分布，q(也就是y.hat)则为训练后的模型的预测标记分布， 交叉熵损失函数可以衡量p与q的相似性，相似性大，说明预测准确
        loss = -np.log(input_tensor + np.finfo(float).eps)
    #     np.finfo.eps(dtype) prevent values close to log(0), das ist sinnlos, here choosen float type
        loss_entropy = np.sum(loss * label_tensor)
        return loss_entropy

    def backward(self, label_tensor):
        # En = -y/y.hat  y is label_tensor, y.hat is input_tensor, our prediction
        error_tensor = -np.divide(label_tensor, self.input_tensor)
        return error_tensor

