import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge_agreement')
    parser.add_argument('--file_path_teacher', type=str, default='/home/xuanli/Knowledge_agreement/data/cifar100_train_kd.txt')
    parser.add_argument('--file_path_student', type=str, default='/home/xuanli/Knowledge_agreement/data/cifar100_train_student.txt')
    parser.add_argument('--percentile', type=float, default=0.05, help='choose top percentile')
    args = parser.parse_args()
    return args


def parse_df(file_path):
    with open(file_path, 'r') as f:
        df = f.readlines()
    df = list(map(lambda x: [x.split(',')[0], int(x.split(',')[1])] + list(map(float, x.strip().split(',')[2].split('_'))), df))
    df = pd.DataFrame(df)
    # _df = df.loc[df.iloc[:, 2:].max(1).argsort()[::-1]]
    # _df = _df.reset_index()
    return df


def compare(df_t, df_s, p):
    keeps = int(len(df_t) * p)
    t = df_t.loc[:keeps, 0].to_numpy()
    s = df_s.loc[:keeps, 0].to_numpy()
    _x = set(t) - set(s)
    print(f'{len(_x)} files in teacher but not in student:')
    _y = set(s) - set(t)
    print(f'{len(_y)} files in student but not in teacher:')
    # Step 1: get teacher, student accuracy
    acc_t = accuracy(df_t.iloc[:, 1], df_t.iloc[:, 2:])
    acc_s = accuracy(df_s.iloc[:, 1], df_s.iloc[:, 2:])
    # general agreement
    pseudo_gt = df_t.iloc[:, 2:].to_numpy().argmax(1)
    acc_agg = accuracy(pseudo_gt, df_s.iloc[:, 2:])
    print(acc_t, acc_s, acc_agg)
    # train
    # 0.99552 0.97084 0.97332
    # Step 2:
    res = agreement(df_t.iloc[:, 2:], df_s.iloc[:, 2:])
    # Find teacher student agreement


def accuracy(gt, acc):
    temp = acc.to_numpy().argmax(1)
    res = (gt==temp).mean()
    return res


def agreement(t, s):
    t_1 = t.to_numpy().argmax(1)
    t_2 = t.to_numpy().max(1)
    print(t_2.shape)





if __name__ == '__main__':
    args = parse_args()
    df_teacher = parse_df(args.file_path_teacher)
    df_student = parse_df(args.file_path_student)
    compare(df_teacher, df_student, args.percentile)



