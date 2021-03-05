from abc import abstractmethod
from collections.abc import Iterable

from aibase import *


class Layer:
    in_shape: tuple
    out_shape: tuple

    def __init__(self, in_shape, out_shape, forward):
        """
        初始化一个层
        :param in_shape: tuple 输入层形状
        :param out_shape: tuple 输出层形状
        :param forward: 前向传播函数
        :var self.value: 正向传播值(a，对于没有激活的层==z)
        """
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.forward = forward
        self.value = 0

    @abstractmethod
    def backward(self, error):
        """
        将误差传播至上一层
        :param error: 本层误差, dE/da
        :return: dE/dx, equals to error_i-1 cause x_i == z_i-1 here.
        """
        pass

    @abstractmethod
    def update(self, error, rate):
        """
        使用本层误差dE/da=error去更新每层出现的所有参数（梯度下降）
        :param error: 本层误差
        :param rate: 学习率
        :return None
        """
        pass


class FullConn(Layer):
    def __init__(self, in_count, out_count, forward='sigmoid'):
        """
        :param forward: 前向传播函数名，决定了此处采用的激活函数
        :var self.activation: 使用的激活函数
        :var self.deriv: 反向传播函数，不需要error参数
        """
        if forward == 'sigmoid':
            self.activation = self.sigmoid
            self.deriv = self.dsig
        elif forward == 'softmax':
            self.activation = self.softmax
            self.deriv = self.dsoftmax
            def set_target(tar):  # 使用装饰器实现添加类函数
                self.target = tar
            self.set_target = set_target
        elif forward == 'rbf':  # 径向基函数
            self.activation = self.rbf
            self.deriv = self.drbf

        if not isinstance(in_count, Iterable):
            in_count = (in_count,)  # 统一转为tuple
        if not isinstance(out_count, Iterable):
            out_count = (out_count,)
        super(FullConn, self).__init__(in_shape=in_count, out_shape=out_count, forward=self.forward)
        self.w = np.random.randn(np.prod(in_count), np.prod(out_count))  # 展开
        if not forward == 'rbf':  # rbf没有偏置
            self.b = np.random.randn(*out_count)

    def forward(self, x):
        return np.reshape(self.activation(np.dot(x.flatten(), self.w) + self.b.flatten()),
                          self.out_shape)  # 先展开做内积再重整

    def sigmoid(self, x):
        self.value = sig(x)
        return self.value

    def dsig(self):
        return self.value * (1 - self.value)  # 这里是broadcasting

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)  # 防止溢出
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        self.value = y
        return y

    def dsoftmax(self):
        return self.value - self.target  # BUT WHY??? TODO

    def rbf(self, x):
        """
        基于欧氏距离的径向基函数
        # TODO: rewrite
        """
        self.value = [np.linalg.norm(x - i, ord=2) for i in self.w]
        return self.value

    def drbf(self, error):
        # TODO:
        return

    def backward(self, error):
        return np.dot(self.w, error.flatten() * self.deriv())

    def update(self, error, rate):
        # TODO:
        pass


class Conv(Layer):
    def __init__(self, in_shape, out_shape, step=1):
        """
        :var self.w: 卷积核
        :var self.b: 偏置
        :parameter step: 步长（默认为1）
        """
        super(Conv, self).__init__(in_shape, out_shape, forward=self.conv)
        self.w = np.random.randn(out_shape[0], 5, 5)  # 有多少个特征图就有多少个卷积核
        self.b = np.random.random()
        self.step = step

    def conv(self, data):
        """
        整个卷积操作
        :param data: 输入的三维张量
        :var self.w: 本层全部卷积核组成的二维张量
        :return: 卷积结果，三维张量
        :assert out_shape[0] = conv_group.shape[0]
        """
        self.data = data
        z_out = np.zeros(self.out_shape)
        for out_index in range(self.out_shape[0]):  # 每张特征图
            for inX in self.conv_group[out_index]:  # 遍历这张特征图对应的输入，inX是当前使用的输入编号
                z_out[out_index] += conv2d(data[inX], self.w[out_index])  # 使用feature map对应的卷积核
        z_out += self.b
        self.value = z_out
        return z_out

    def dconv(self, error):
        # TODO: check
        error_1 = np.zeros(self.in_shape)
        for out_index in range(self.out_shape[0]):  # 每张特征图
            for inX in self.conv_group[out_index]:  # 遍历这张特征图对应的输入，inX是当前使用的输入编号
                error_1[inX] += self.dconv2d(error[out_index])
        return error_1

    def dconv2d(self, error):
        return conv2d(error, rot180_2d(self.w))

    def set_combination(self, group):
        """
        带有组合的卷积层
        :param group: 卷积组，2-d数组
        """
        self.conv_group = group

    def backward(self, error):
        return self.dconv(error)

    def update(self, error, rate):
        error = np.reshape(error, self.out_shape)
        self.w += rate * error * self.data  # TODO:
        self.b += rate * np.sum(error)


class MaxPool(Layer):
    def __init__(self, in_shape, out_shape, window=(2, 2)):
        """
        :param window: 池化窗口，默认为2
        这里默认step==window.size
        """
        super(MaxPool, self).__init__(in_shape, out_shape, self.maxpooling)
        self.window = window

    def maxpooling(self, input_img):
        """
        :param input_img: 三维张量
        :var self.max_index: pair<int, int> array[picture_number][coordinate_index]
        :return: 池化结果
        """
        pic, x, y = input_img.shape
        wx, wy = self.window
        out_pooling = np.zeros((pic, x // wx, y // wy))
        self.max_index = list()
        for p in range(pic):
            tmp = list()
            for i in range(0, y // wy):
                for j in range(0, x // wx):
                    out_pooling[p, i, j] = np.max(input_img[p, i * wy: i * wy + wy, j * wx: j * wx + wx])
                    # 记录每个池化窗孔最大值所在的下标，以供反向传播error
                    _idx = np.argmax(input_img[p, i * wy: i * wy + wy, j * wx: j * wx + wx])
                    tmp.append((_idx // wx + i * wy, _idx % wx + j * wx))
            self.max_index.append(tmp)
        self.value = out_pooling
        return out_pooling

    def dmaxpool(self, error):
        error = np.reshape(error, self.out_shape)
        wx, wy = self.window
        error_1 = np.tile(np.zeros_like(error), self.window)
        for p in range(self.in_shape[0]):  # picture numbers
            for i in self.max_index[p]:
                x = i[0]
                y = i[1]
                error_1[p, x, y] = error[p, x // wx, y // wy]
        return error_1

    def backward(self, error):
        return self.dmaxpool(error)

    def update(self, error, rate):
        # 池化层没有参数需要更新
        pass


class Network:
    def __init__(self, sequence, learn_rate):
        self.layers = sequence
        self.rate = learn_rate

    def load(self, data, label):
        self.t_data = data
        self.t_label = label

    def forward(self):
        idx = np.random.randint(0, self.t_data.shape[0])  # 使用SGD，随机产生本次使用的数据编号
        print('choose', idx)
        a = self.t_data[idx]
        a = np.expand_dims(a, 0)  # expand to (1, 32, 32)
        ans = self.t_label[idx]
        label = np.zeros(10)
        label[ans] = 1  # 产生one-hot vector作为真实label
        for l in self.layers:
            a = l.forward(a)  # 通过每一层的正向传播函数，最终得到predict label
        return a, label

    def backprop(self, error):
        """

        :param error: target-output 为dE/da(output)，需对输出层求导方可得到所需dE/dz
        """
        for l in reversed(self.layers):
            l.update(error, self.rate)  # 更新本层参数，也就是使用dE/dz更新dE/dw等
            error = l.backward(error)  # 求dz_l/dz_(l-1)，将误差传播到上一层


def run():
    # 定义网络结构
    lys = list()
    lys.append(Conv((1, 32, 32), (6, 28, 28)))  # C1
    lys[-1].set_combination(np.zeros((6, 1)).astype(np.int32))  # 每组单张卷积核
    lys.append(FullConn((6, 28, 28), (6, 28, 28)))  # sigmoid -- 这里将卷积和之后的激活视作两层
    lys.append(MaxPool((6, 28, 28), (6, 14, 14)))  # S2
    lys.append(Conv((6, 14, 14), (16, 10, 10)))  # C3
    lys[-1].set_combination(
        [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 4, 5], [0, 1, 5], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5],
         [0, 3, 4, 5],
         [0, 1, 4, 5], [0, 1, 2, 5], [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5], [0, 1, 2, 3, 4, 5]])  # 卷积核组
    lys.append(FullConn((16, 10, 10), (16, 10, 10)))  # sigmoid
    lys.append(MaxPool((16, 10, 10), (16, 5, 5)))  # S4
    lys.append(FullConn((16, 5, 5), 120))  # C5 - full connection
    lys.append(FullConn(120, 84))  # full connection
    lys.append(FullConn(84, 10, 'softmax'))  # output - rbf function
    lenet = Network(sequence=lys, learn_rate=1e-5)
    # 载入训练数据
    (train, train_label), (test, test_label) = get_mnist_data()
    lenet.load(train, train_label)
    # 单次训练
    output, target = lenet.forward()
    E = np.linalg.norm(output - target, ord=2)  # 平方损失
    print('MSE lose =', E)
    # 反向传播
    lenet.layers[-1].set_target(target)  # softmax求导时会用到target
    lenet.backprop(target - output)


run()
