import numpy as np
import matplotlib.pyplot as plt


def case_do_every_element():
    """
    将读入多个数据全部进行某种操作（如整型化）
    第一个方法是list，第二个方法是通过映射函数
    """
    a, b, c = [int(i) for i in input().split()[:3]]
    print(a, b, c, type(a))
    a, b, c = list(map(int, input().split()[:3]))
    print(a, b, c, type(a))
    a, *b, c = input().split(',')  # 序列解包


def case_sum():
    """
    对数组采用不同方式求和
    """
    arr = np.arange(20).reshape(4, 5)
    print(arr)
    print(np.sum(arr[:, 0:2]))  # 求0-1列的和
    print(np.sum(arr[:, 0:2], axis=0))  # 分别求0列和1列的和， axis=0为纵轴
    print(np.sum(arr[:, 0:2], axis=1))  # 对于0-1列分别求每行的和， axis=1为横轴


def case_numpy_and_math_type():
    """
    numpy和math都是C实现的，有大小限制，可以指定数据类型
    """
    a = np.array([10000000000000], dtype='int64')
    print(a)


def case_show_gray_img():
    a = np.array(range(100))
    k = [1, 2]
    img = np.outer(k, a)
    plt.imshow(img, cmap='gray')
    plt.show()


def case_np_array():
    """
    numpy的array类型必须固定，可以自动向上兼容
    """
    cost = list()
    for i in range(0, 10):
        cost.append(i)
    cost = np.array(cost)
    cost = cost.reshape(1, 5, 2)
    print(cost)
    print(np.append(cost, cost, axis=0))
    print(np.append(cost, cost, axis=1))
    print(np.append(cost, cost, axis=2))


def case_array_multiply():
    a = b = np.arange(1, 10).reshape(3, 3)
    print(a)
    print(np.prod(a))  # 所有元素乘积

    print(a * b)
    print(np.multiply(a, b))
    # 对应元素相乘，可以广播（数量积）

    print(np.matmul(a, b))
    print(np.dot(a, b))  # 若其中一个为标量，则为数量积
    print(a @ b)
    # 矩阵乘，不能广播（矢量积）

    print(b[0], np.dot(a, b[0]))
    print(b[:, 0], np.dot(a, b[:, 0]))

    print(np.inner(a, b))  # 内积 矩阵时result[i, j] = a[i,:]*b[j,:]的各元素之和
    print(np.outer(a, b))  # 外积 克特罗内积


def case_multiarg_for_zip():
    # zip将多个列表打包为多个元组的列表
    a = range(1, 10)
    b = reversed(a)
    for i, j in zip(a, b):
        print(i, j)


def case_array_choose():
    # 数组筛选操作
    a = np.arange(0, 10).reshape(2, 5)
    b = np.where(a > 3)
    for i, j in zip(b[0], b[1]):
        print(a[i][j])  # list是嵌套列表而非高维数组，不能用元组索引，而是索引展开后再索引
    print(np.extract(a > 3, a))


def case_array_shape():
    # 要注意shape完全匹配或者可以广播才行
    a = np.arange(0, 10)
    print(a.shape)  # 长度为10的一维数组 shape=(10,)
    a = a.reshape(1, 10)  # 1行10列的二维数组 等价于  a = np.expand_dims(a, axis = 0)
    print("a = ", a)
    print("append and expand:", np.append(a, np.arange(10, 20)))  # axis为None时总是自动展开成一维数组
    print("append from bottom:\n", np.append(a, np.arange(10, 20).reshape(1, 10), axis=0))  # 必须reshape成同样规格才可以
    print("append from right:\n", np.append(a, np.arange(10, 20).reshape(1, 10), axis=1))
    print("broadcasting:\n", np.append(
        a,
        np.broadcast_to(np.arange(10, 15).reshape(1, 5), (1, 10)),
        axis=0
    ))


def case_broadcasting():
    # broadcast相关函数很多
    # 广播规则：若两argument有某一共同维度，则它们必须相同或其一者为1
    a = np.arange(1, 6)
    b = np.broadcast_to(a, (3, 5))
    print(b)
    print(a + np.arange(1, 16).reshape(3, 5))


def case_expand_array():
    # repeat : 将每个元素都拓展
    # tile : 将拓展数组本身
    # expand_dims : 拓展出新的轴
    a = np.array([1, 2, 3, 4, 5])
    b = np.expand_dims(a, axis=0)  # 0为原本是横列拓展纵列，1为将原本视为纵列拓展横列
    print(b)
    print(b.repeat(5, axis=0))
    print(b.repeat(3, axis=1))  # 必须在原有的轴上拓展
    print(np.tile(b, (2, 2)))  # 构造2*2个b


def case_args_kwargs():
    """
    kwargs接受的是多个参数而不是一个dict，所以如果是dict就要进行解包：**dict, *tuple
    """
    def func(*args, **kwargs):  # 参数关键字只能接受string
        for x in args:
            print(x)
        for l in kwargs:
            print(l, kwargs[l])
        if 'c' in kwargs:
            print('found c')
        else:
            print('not found c')
    a = {'a': 143, 'b': 'xyz', 'd': 1122, '4': 5}
    func(**a)  # 将a解包
    func(input=1)  # kwargs必须写明key
    func(123)  # 不写明的进入args
    # func(input=1, 123) 错误！args必须先于kwargs


def case_unpack():
    def f():
        return 1, 2, 3
    print(*f())  # 将(1,2,3)变为1,2,3
    *a, = [1, 2, 3]  # a获得(1,2,3)

