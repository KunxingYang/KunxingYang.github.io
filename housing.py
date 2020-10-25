import numpy as np
import json

# 全局参数
# 训练集比例
TRAIN_RATIO = 0.8

# 数据集路径
DATA_FILE_PATH='./data/housing.data'

def load_data(file_name=DATA_FILE_PATH):
    data_file = file_name
    data = np.fromfile(data_file, sep=' ')

    # data包含了所有的数据，每一条数据包括13个特征值及一个真值
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]

    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 切分数据集,数据集偏移量
    offset = int(data.shape[0] * TRAIN_RATIO)
    training_data = data[:offset]

    # 数据归一化，都归一化到0-1
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), (training_data.sum(axis=0) / training_data.shape[0])
    for i in range(feature_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

class Network(object):
    def __init__(self, num_of_weights):
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights)
        self.b = 0

    def forward(self, x):
        return np.dot(x, self.w) + self.b

    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost

if __name__ == "__main__":
    # 获取数据
    training_data, test_data = load_data()
    x = training_data[:, :-1]
    y = training_data[:, -1:]

    net = Network(13)
    x1 = x[0:3]
    y1 = y[0:3]
    z = net.forward(x1)
    print("predict: ", z)
    loss = net.loss(z, y1)
    print('loss: ', loss)

