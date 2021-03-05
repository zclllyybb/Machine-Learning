import numpy as np
import math
import os
from PIL import Image
from scipy import io as sio
from keras.datasets import mnist


def get_mnist_data():
    (train, train_label), (test, test_label) = mnist.load_data()
    # 28*28 padding to 32*32
    form_train = np.zeros((60000, 32, 32))
    form_test = np.zeros((10000, 32, 32))
    for i in range(len(train)):
        form_train[i, 2:30, 2:30] = train[i]
    for i in range(len(test)):
        form_test[i, 2:30, 2:30] = test[i]
    return (form_train, train_label), (form_test, test_label)


def get_train_pattern():
    # 返回训练集的特征和标签
    current_dir = os.getcwd() + "/"
    train = sio.loadmat(current_dir + "mnist_train.mat")["mnist_train"]
    train_label = sio.loadmat(
        current_dir + "mnist_train_labels.mat")["mnist_train_labels"]
    train_label = train_label.flatten()
    train = (train - np.min(train)) / (np.max(train) - np.min(train))  # 归一化
    return train, train_label


def get_test_pattern():
    # 返回测试集
    base_url = os.getcwd() + "/mnist_test/"
    test_img_pattern = []
    for i in range(10):
        img_url = os.listdir(base_url + str(i))
        t = []
        for url in img_url:
            img = Image.open(base_url + str(i) + "/" + url)
            img = img.convert('L')  # 归一化
            img_array = np.asarray(img, 'i')  # 转化为int数组
            img_vector = img_array.reshape(
                img_array.shape[0] * img_array.shape[1])  # 展开成一维数组
            t.append(img_vector)
        test_img_pattern.append(t)
    return test_img_pattern


def __sig(gamma):
    if gamma < 0:  # 这一判断不能矢量化，因此__sig是非矢量化的函数
        return 1 - 1 / (1 + math.exp(gamma))
    else:
        return 1 / (1 + math.exp(-gamma))  # gamma < 0时e^x结果过大导致溢出math range error


sig = np.vectorize(__sig)  # 将__sig矢量化


def conv2d(data, w):
    """
    单张输入和单个卷积核进行卷积
    :param data: 二维输入图片
    :param w: 二维卷积核，默认使用本对象卷积核
    :return: 一张特征图，二维张量
    """
    n, m = data.shape
    wx, wy = w.shape
    img_new = np.zeros((n - wx + 1, m - wy + 1))
    for i in range(n - wx + 1):
        for j in range(m - wy + 1):
            a = data[i:i + wx, j:j + wy]
            img_new[i][j] = np.sum(np.multiply(w, a))  # 卷积区域和卷积核相乘
    return img_new


def rot180_2d(conv_filters):
    return np.flipud(np.fliplr(conv_filters))

