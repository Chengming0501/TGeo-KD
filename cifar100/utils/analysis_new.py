import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge_agreement')
    parser.add_argument('--file_path_teacher', type=str, default='/home/xuanli/Knowledge_agreement/data/cifar100_val_kd.txt')
    parser.add_argument('--file_path_student', type=str, default='/home/xuanli/Knowledge_agreement/data/cifar100_val_student.txt')
    args = parser.parse_args()
    return args


def parse_df(file_path):
    with open(file_path, 'r') as f:
        df = f.readlines()
    df = list(map(lambda x: [x.split(',')[0], int(x.split(',')[1])] + list(map(float, x.strip().split(',')[2].split('_'))), df))
    df = pd.DataFrame(df)
    return df


def compare(df_t, df_s):
    print(type(df_t))
    temp = df_t.loc[:, 0] == df_s.loc[:, 0]
    assert temp.mean() == 1, 'name does not match'
    acc_t = accuracy(df_t.iloc[:, 1], df_t.iloc[:, 2:])
    acc_s = accuracy(df_s.iloc[:, 1], df_s.iloc[:, 2:])
    conf_t = confidence(df_t.loc[:, 1], df_t.loc[:, 2:])
    conf_s = confidence(df_s.loc[:, 1], df_s.loc[:, 2:])
    agree_t = agreement(df_t.loc[:, 1], df_t.loc[:, 2:], df_s.loc[:, 2:])
    print(f'accuracy: teacher {acc_t}, student {acc_s}')
    print(f'conf max t: {conf_t.max()}, conf max s: {conf_s.max()}')
    print(f'conf min t: {conf_t.min()}, conf min s: {conf_s.min()}')
    # plt.scatter(conf_t, agree_t)

    plt.scatter(conf_t, conf_s)
    plt.show()
"""
undirectional:
1. for train
conf_s.max == 0.4870
conf_t.max = 0.9999

conf_s.min = 4e-7
conf_t.min = 0.0014

2.
"""


def confidence(gt, prob, direction=False):
    _gt = torch.from_numpy(gt.to_numpy())
    _prob = torch.from_numpy(prob.to_numpy())
    val, ind = _prob.topk(2, dim=1)
    if direction:
        pass
    else:
        conf = val[:, 0] - val[:, 1]
    return conf.numpy()


def agreement(gt, t, s, direction=False):
    _gt = torch.from_numpy(gt.to_numpy())
    _t = torch.from_numpy(t.to_numpy())
    _s = torch.from_numpy(s.to_numpy())
    if direction:
        pass
    else:
        _v_t, _ind_t = _t.max(1)
        _v_s, _ind_s = _s.max(1)
        agree = _ind_t == _ind_s
    return agree.numpy()


def accuracy(gt, acc):
    temp = acc.to_numpy().argmax(1)
    res = (gt == temp).mean()
    return res


if __name__ == '__main__':
    args = parse_args()
    df_teacher = parse_df(args.file_path_teacher)
    df_student = parse_df(args.file_path_student)
    compare(df_teacher, df_student)



