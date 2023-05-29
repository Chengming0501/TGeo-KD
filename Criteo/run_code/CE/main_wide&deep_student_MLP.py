import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import numpy as np
from model.Wide_Deep_CE import Wide_Deep, Controller
from argparse import ArgumentParser
from utils import EarlyStopping
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from torchmetrics.classification import BinaryAccuracy
from helper.evaluation import compute_metrics
from torchmetrics import Accuracy
from prettytable import PrettyTable

from imblearn.over_sampling import RandomOverSampler

# define the oversampling object
oversampler = RandomOverSampler()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parse_args():
    parser = ArgumentParser(description="KD")
    parser.add_argument("--data_name", type=str, default="criteo")

    # parser.add_argument('--val_ratio', type=float, default=0.1)
    # parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--is_logging', type=bool, default=False)
    # Seed
    parser.add_argument('--seed', type=int, default=2023, help="Seed (For reproducability)")
    # Model
    parser.add_argument("--model_name", type=str, default="Wide_Deep")
    parser.add_argument('--dim', type=int, default=16, help="Dimension for embedding")
    parser.add_argument('--dnn_hidden_units', nargs='+', type=int, default=[64, 32], help='hidden layer dimensions')
    parser.add_argument('--alpha', type=float, default=0.7, help="trade-off on teacher")
    parser.add_argument('--temp', type=float, default=2.0, help="temperature")
    parser.add_argument('--state_type', type=int, default=1, help="state")
    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--wd', type=float, default=1e-2, help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs', type=int, default=1000, help="Number of epoch during training")
    parser.add_argument('--every', type=int, default=1,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--patience', type=int, default=20, help="patience")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch_size")

    return parser.parse_args()


def onehot_matrix(binary_vector):
    num_rows = len(binary_vector)
    one_hot_matrix = np.zeros((num_rows, 2))
    one_hot_matrix[np.arange(num_rows), binary_vector] = 1
    return torch.tensor(one_hot_matrix).float()

def generate_controller_state(value_s, value_t, value_g, value_o, state_type):
    if state_type == 1:
        return torch.cat((
            value_s, value_t, value_g,
        ), dim=1)
    elif state_type == 2:
        return torch.cat((
            torch.pow(value_s - value_t, 2),
            torch.pow(value_t - value_g, 2),
            torch.pow(value_g - value_s, 2),
        ), dim=1)
    elif state_type == 3:
        return torch.cat((
            value_s, value_t, value_g,
            torch.pow(value_s - value_t, 2),
            torch.pow(value_t - value_g, 2),
            torch.pow(value_g - value_s, 2),
        ), dim=1)

def train(args):
    os.chdir('/Users/../Research_project/KD_tradeoff/')
    print(os.getcwd())
    path = 'data_raw/{}/'.format(args.data_name)
    if args.data_name == 'criteo':
        sparse_feature = ['C' + str(i) for i in range(1, 27)]
        dense_feature = ['I' + str(i) for i in range(1, 14)]
        col_names = ['label'] + dense_feature + sparse_feature
        print("col_names:", col_names)
        df = pd.read_csv(path + 'dac_sample.txt', names=col_names, sep='\t')

        df[sparse_feature] = df[sparse_feature].fillna('-1', )
        df[dense_feature] = df[dense_feature].fillna('0', )

        feat_sizes = {}
        feat_sizes_dense = {feat: 1 for feat in dense_feature}
        feat_sizes_sparse = {feat: len(df[feat].unique()) for feat in sparse_feature}
        feat_sizes.update(feat_sizes_dense)
        feat_sizes.update(feat_sizes_sparse)

        for feat in sparse_feature:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])
        nms = MinMaxScaler(feature_range=(0, 1))
        df[dense_feature] = nms.fit_transform(df[dense_feature])

        fixlen_feature_columns = [(feat, 'sparse') for feat in sparse_feature] + [(feat, 'dense') for feat in
                                                                                  dense_feature]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        x, y = df.iloc[:, 1:], df.iloc[:, 0]

    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=args.seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.1,
                                                      random_state=args.seed)

    x_train_t, x_train_s, y_train_t, y_train_s = train_test_split(x_train, y_train, test_size=0.3,
                                                                  random_state=args.seed)
    x_val_t, x_val_s, y_val_t, y_val_s = train_test_split(x_val, y_val, test_size=0.3,
                                                          random_state=args.seed)

    x_train, y_train = oversampler.fit_resample(x_train_s, y_train_s)
    x_val, y_val = x_val_s, y_val_s

    # x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=args.seed)

    print("y_train:", len(y_train), y_train.sum())
    print("y_val:", len(y_val), y_val.sum())
    print("y_test:", len(y_test), y_test.sum())
    # np.savetxt('./saved/{}/gt_label_train.txt'.format(args.data_name), np.array(y_train.to_numpy()), delimiter=',')
    # np.savetxt('./saved/{}/gt_label_val.txt'.format(args.data_name), np.array(y_val.to_numpy()), delimiter=',')
    # np.savetxt('./saved/{}/gt_label_test.txt'.format(args.data_name), np.array(y_test.to_numpy()), delimiter=',')

    print("x_train shape:{}, y_train shape{}".format(x_train.shape, y_train.shape))
    print("x_val shape:{}, y_val shape{}".format(x_val.shape, y_val.shape))
    print("x_test shape:{}, y_test shape{}".format(x_test.shape, y_test.shape))

    train_dataset_raw = TensorDataset(torch.from_numpy(np.array(x_train)), torch.from_numpy(np.array(y_train)))
    val_dataset = TensorDataset(torch.from_numpy(np.array(x_val)), torch.from_numpy(np.array(y_val)))
    test_dataset = TensorDataset(torch.from_numpy(np.array(x_test)), torch.from_numpy(np.array(y_test)))

    train_dataloader_raw = DataLoader(train_dataset_raw, shuffle=False, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    # load teacher logits
    teacher_dnn_hidden_units = [64, 64]
    hidden_units_str = '_'.join(map(str, teacher_dnn_hidden_units))
    model_teacher = Wide_Deep(feat_sizes, args.dim, linear_feature_columns, dnn_feature_columns,
                              teacher_dnn_hidden_units).to(
        args.device)
    saved_teacher_path = './saved/{}/model_{}/teacher_dim{}_CE_{}.pt'.format(args.data_name, args.model_name, args.dim,
                                                                             hidden_units_str)
    model_teacher.load_state_dict(torch.load(saved_teacher_path))
    true_label = []
    teacher_logits = []
    for x, y in train_dataloader_raw:
        x, y = x.to(args.device).float(), y.to(args.device).long()
        y_pre = model_teacher(x)
        true_label.extend(y.tolist())
        teacher_logits.extend(y_pre.tolist())

    print("true_label:", np.shape(true_label))
    print("teacher_logits:", np.shape(teacher_logits))


    train_dataset = TensorDataset(torch.from_numpy(np.array(x_train)), torch.from_numpy(np.array(y_train)),
                                  torch.from_numpy(np.array(teacher_logits)))
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    # num_users, num_items = max(df[0]) + 1, max(df[1]) + 1
    print("args.dnn_hidden_units:", args.dnn_hidden_units)
    if args.model_name == 'Wide_Deep':
        model = Wide_Deep(feat_sizes, args.dim, linear_feature_columns, dnn_feature_columns, args.dnn_hidden_units).to(
            args.device)

    if args.state_type in [1, 2]:
        in_dim = 6
    elif args.state_type in [3]:
        in_dim = 12
    controller = Controller(in_dim, args.device).to(args.device)

    model_optimizer = torch.optim.Adam(model.parameters(),
                                       lr=args.lr,
                                       weight_decay=args.wd)
    controller_optimizer = torch.optim.Adam(controller.parameters(),
                                            lr=args.lr,
                                            weight_decay=args.wd)

    loss_func = torch.nn.CrossEntropyLoss(reduction='none').to(args.device)

    saved_path = './saved/{}/model_{}/student_dim{}_CE_state{}.pt'.format(args.data_name, args.model_name, args.dim,
                                                                       args.state)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=saved_path)

    for epoch in range(args.n_epochs):
        model.train()
        total_loss, total_len = 0, 0
        total_GT_loss, total_KD_loss = 0, 0
        for x, y_g, y_t in train_dataloader:
            x, y_g, y_t = x.to(args.device).float(), y_g.to(args.device).long(), y_t.to(
                args.device).float()

            y_pre = model(x)

            value_s = F.softmax(y_pre / args.temp, dim=1)
            value_t = F.softmax(y_t / args.temp, dim=1)
            value_g = onehot_matrix(y_g)
            value_o = value_t

            # print("value_s:", value_s[:5])
            # print("value_t:", value_t[:5])
            # print("value_g:", value_g[:5])

            state = generate_controller_state(value_s, value_t, value_g, value_o, args.state_type)

            trade_off = controller(state).squeeze()
            loss_GT = loss_func(y_pre, y_g)
            loss_KD = F.kl_div(F.log_softmax(y_pre / args.temp, dim=1), F.softmax(y_t / args.temp, dim=1),
                               reduction='none').sum(1)
            loss = (1 - trade_off) * loss_GT + trade_off * args.temp ** 2 * loss_KD
            loss = torch.mean(loss)

            model_optimizer.zero_grad()
            controller_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            controller_optimizer.step()

            total_loss += loss.item() * len(y_g)
            total_GT_loss += loss_GT.mean().item() * len(y_g)
            total_KD_loss += loss_KD.mean().item() * len(y_g)
            total_len += len(y_g)
        train_loss = total_loss / total_len
        total_GT_loss = total_GT_loss / total_len
        total_KD_loss = total_KD_loss / total_len
        print("epoch {}, train loss is {:.6f}".format(epoch, train_loss))
        print("fuion ratio:", torch.max(trade_off), torch.mean(trade_off), torch.min(trade_off))
        # print("train_loss:", train_loss)
        # print("total_GT_loss:", total_GT_loss)
        # print("total_KD_loss:", total_KD_loss)
        print("-------------------------------")

        if epoch % args.every == 0:
            model.eval()
            with torch.no_grad():
                valid_acc, valid_auc, valid_nll = compute_metrics(model, val_dataloader, args.device)
                early_stopping(valid_auc, model)

                if early_stopping.early_stop:
                    print("Early stopping")

                    break

    model.load_state_dict(torch.load(saved_path))
    acc, auc, nll = compute_metrics(model, test_dataloader, args.device)

    return acc, auc, nll


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", args.device)

    state_list = [1, 2, 3]
    dnn_hidden_units_list = [
        [64, 64]
    ]
    result_table = PrettyTable(['Dataset', 'state', 'hidden_units', 'acc', 'auc', 'nll'])
    for state in state_list:
        for dnn_hidden_units in dnn_hidden_units_list:
            args.state = state
            args.dnn_hidden_units = dnn_hidden_units
            acc, auc, nll = train(args)
            result_table.add_row([args.data_name, args.state, dnn_hidden_units,  acc, auc, nll])

    print(result_table)
