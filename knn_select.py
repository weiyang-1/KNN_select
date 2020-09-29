# -*- coding: utf-8 -*-

"""
@Created: 2020/9/27 17:38
@AUTH: MeLeQ
@Used: pass
"""


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def KNN_S():
    """
    通过A品牌与B品牌映射关系，训练出A--->B的唯一对应关系
    :param x:
    :return:
    """

    # 获取数据源
    data_train = pd.read_csv('./data/train_data.csv', sep=" ")
    data_test = pd.read_csv('./data/test_data.csv', sep=" ")

    # 区分训练集合和测试集合
    train_x = [[i] for i in data_train.brandA]
    test_x = [[i] for i in data_test.brandA]
    train_y = np.array(data_train.brandB)
    test_y = np.array(data_test.brandB)

    # 测试不同的参数选择得到的结果差异
    n_neighbors_set_list = range(5, 18)
    score_list = []

    for n_b in n_neighbors_set_list:
        # 训练模型
        model = KNeighborsClassifier(n_neighbors=n_b)
        model.fit(train_x, train_y)
        print(model.predict(test_x))
        score = model.score(test_x, test_y)
        print(score)
        score_list.append(score)

    # 绘制参数和结果图
    plt.plot(n_neighbors_set_list, score_list)
    plt.show()

    # 选择最大的准确率对应的参数训练模型并且保存
    max_score_index_list = [i for i, j in enumerate(score_list) if j == max(score_list)]
    suitable_params = n_neighbors_set_list[max_score_index_list[1]] if len(max_score_index_list)>1 else n_neighbors_set_list[max_score_index_list[0]]
    print(f"最合适的参数为： {suitable_params}")

    # 以最适合的参数训练并保存模型
    model = KNeighborsClassifier(n_neighbors=suitable_params)
    model.fit(train_x, train_y)
    with open("./knn_select.model", "wb") as f:
         pickle.dump(model, f)
    print(f"保存模型成功！")


def pre_data():
    """
    预测数据
    :return:
    """
    with open("./knn_select.model", "rb") as f:
        model = pickle.load(f)
    data_test = pd.read_csv('./data/test_data.csv', sep=" ")
    test_x = [[i] for i in data_test.brandA]
    test_y = np.array(data_test.brandB)
    r = model.predict(test_x)
    print(r)
    print(model.score(test_x, test_y))


if __name__ == '__main__':
    # KNN_S()
    pre_data()