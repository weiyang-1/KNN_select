# -*- coding: utf-8 -*-

"""
@Created: 2020/9/28 15:45
@AUTH: MeLeQ
@Used: pass
"""

FLAG_STR = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def gene_train_data():
    """
    生成训练数据: 假设A品牌映射为数字，B品牌映射为字母
    brandA brandB
    1 A
    2 B
    3 C
    4 D
    5 C
    .
    .
    .
    26 Z
    :return:
    """
    with open('./data/train_data.csv', mode='a', encoding='utf-8') as f:
        for j in range(len(FLAG_STR)):
            # 对给个字母给的个数都是起始位100个  后面递减
            # 即 1 A 初选
            for i in range(j, len(FLAG_STR)):
                for _ in range(100-i):
                    # 保存到训练数据集
                    f.write(str(j+1) + " " + FLAG_STR[i]+"\n")


if __name__ == '__main__':
    gene_train_data()

