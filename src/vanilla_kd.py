import os
import torch
from network import Student, Teacher
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
from argparse import ArgumentParser
from collections import Counter
import matplotlib.pyplot as plt




def parse_args():
    parser = ArgumentParser(description="vanilla_kd")
    parser.add_argument('--num_samples_tea', type=int, default=50000, help="numbers of sample for training teacher")
    parser.add_argument('--num_samples_stu', type=int, default=10000, help="numbers of sample for training student")
    parser.add_argument('--seed', type=int, default=0, help="random state for spitting data (for reproducability)")
    parser.add_argument('--temperature', type=float, default=4.0, help="temperature")
    parser.add_argument('--alpha', type=float, default=0.2, help="trade-off weight on knowledge distillation")
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate for student and teacher training")
    parser.add_argument('--num_epochs_tea', type=int, default=4000, help="number of epoch for training teacher")
    parser.add_argument('--num_epochs_stu', type=int, default=3000, help="number of epoch for training student")
    parser.add_argument('--check', type=int, default=10, help="epoch of check point")
    parser.add_argument('--early_stop', type=float, default=1e-5, help="early stopping criteria")
    return parser.parse_args()



def data_loader(data, num_samples):
    labels = data.iloc[:, -1].to_numpy()
    features = data.iloc[:, :-1].to_numpy()
    features = minMax.fit_transform(features)     # normalization
    x_train, test_features, y_train, test_labels = train_test_split(features, labels, test_size=args.test_ratio,
                                                                                random_state=args.seed)
    train_size = num_samples - int(num_samples * args.valid_ratio)
    train_features = x_train[:train_size, :]
    train_labels = y_train[:train_size]
    valid_features = x_train[train_size:num_samples, :]
    valid_labels = y_train[train_size:num_samples]
    train_features, train_labels = sampler.fit_resample(train_features, train_labels)    # oversampling for training set

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels



def teacher_train(data):
    train_features, train_labels, valid_features, valid_labels, test_features, test_labels = data_loader(data, args.num_samples_tea)
    [m, n] = np.shape(train_features)
    teacher = Teacher(input_dim=n)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    valid_loss_list = []

    # train the teacher network
    for ep in range(args.num_epochs_tea):
        _, train_pred = teacher(torch.from_numpy(train_features).float())
        train_loss = loss_func(train_pred, torch.from_numpy(train_labels))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print("The " + str(ep) + "-th epoch training loss: " + str(train_loss))

        # early stopping checkpoint
        with torch.no_grad():
            _, valid_pred = teacher(torch.from_numpy(valid_features).float())
            valid_loss = loss_func(valid_pred, torch.from_numpy(valid_labels))
            print("The " + str(ep) + "-th epoch validation loss: " + str(valid_loss))
            valid_loss_list.append(valid_loss)
            # check early stopping criteria
            if ep >= args.check:
                valid_loss_check = np.array(valid_loss_list[ep-args.check: ep])
                if all(i <= args.early_stop for i in np.absolute(np.diff(valid_loss_check))):
                    print("Early stopping")
                    break
    torch.save(teacher, "../result/bilevel/teacher.pkl")      # save the teacher network

    return teacher



def student_train(data):
    train_features, train_labels, valid_features, valid_labels, test_features, test_labels = data_loader(data, args.num_samples_stu)
    [m, n] = np.shape(train_features)
    student = Student(input_dim=n)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    teacher = torch.load("../result/teacher.pkl")  # load teacher network
    valid_loss_list = []

    # Train the student network
    for ep in range(args.num_epochs_stu):
        student.train()
        student_knowledge, student_pred_prob = student(torch.from_numpy(train_features).float())
        teacher_knowledge, teacher_pred_prob = teacher(torch.from_numpy(train_features).float())
        student_knowledge_soft = nn.functional.log_softmax(student_knowledge / args.temperature, dim=1)
        teacher_knowledge_soft = nn.functional.softmax(teacher_knowledge / args.temperature, dim=1)
        teacher_pred_label = torch.argmax(teacher_knowledge_soft, dim=1)
        kd_loss = kl_loss(student_knowledge_soft, teacher_knowledge_soft)     # knowledge distillation loss
        gt_loss = loss_func(student_pred_prob, torch.from_numpy(train_labels))     # ground truth loss
        print(kd_loss)
        print(gt_loss)
        loss = args.alpha * kd_loss + (1 - args.alpha) * gt_loss     # total loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("The " + str(ep) + "-th epoch training loss: " + str(loss))

        student.eval()
        with torch.no_grad():
            _, valid_pred = student(torch.from_numpy(valid_features).float())
            valid_loss = loss_func(valid_pred, torch.from_numpy(valid_labels))
            print("The " + str(ep) + "-th epoch validation loss: " + str(valid_loss))
            valid_loss_list.append(valid_loss)
            # check early stopping criteria
            if ep >= args.check:
                valid_loss_check = np.array(valid_loss_list[ep - args.check: ep])
                if all(i <= args.early_stop for i in np.absolute(np.diff(valid_loss_check))):
                    print("Early stopping")
                    break

    torch.save(student, "../result/vanilla_kd_student.pkl")  # save the student network

    return student



def teacher_test(data):
    acc = MulticlassAccuracy(num_classes=args.num_classes, average=None, top_k=2)
    acc_avg = MulticlassAccuracy(num_classes=args.num_classes, average="micro", top_k=2)
    auc = MulticlassAUROC(num_classes=args.num_classes, average=None)
    auc_avg = MulticlassAUROC(num_classes=args.num_classes, average="macro")
    pre = MulticlassPrecision(num_classes=args.num_classes, average=None)
    pre_avg = MulticlassPrecision(num_classes=args.num_classes, average="micro")
    f1 = MulticlassF1Score(num_classes=args.num_classes, average=None)
    f1_avg = MulticlassF1Score(num_classes=args.num_classes, average="micro")

    _, _, _, _, test_features, test_labels = data_loader(data, args.num_samples_stu)
    teacher = torch.load("../result/bilevel/teacher.pkl")        # load teacher network
    _, test_pred = teacher(torch.from_numpy(test_features).float())
    print("Testing accuracy in each class:{}".format(acc(test_pred, torch.from_numpy(test_labels))))
    print("Average testing accuracy:{}".format(acc_avg(test_pred, torch.from_numpy(test_labels))))
    print("Testing AUC score in each class:{}".format(auc(test_pred, torch.from_numpy(test_labels))))
    print("Average testing AUC score:{}".format(auc_avg(test_pred, torch.from_numpy(test_labels))))
    print("Testing cross entropy loss:{}".format(loss_func(test_pred, torch.from_numpy(test_labels))))
    print("Testing precision in each class:{}".format(pre(test_pred, torch.from_numpy(test_labels))))
    print("Average testing precision:{}".format(pre_avg(test_pred, torch.from_numpy(test_labels))))
    print("Testing F1 score in each class:{}".format(f1(test_pred, torch.from_numpy(test_labels))))
    print("Average testing F1 score in each class:{}".format(f1_avg(test_pred, torch.from_numpy(test_labels))))



def student_test(data):
    acc = MulticlassAccuracy(num_classes=args.num_classes, average=None, top_k=2)
    acc_avg = MulticlassAccuracy(num_classes=args.num_classes, average="micro", top_k=2)
    auc = MulticlassAUROC(num_classes=args.num_classes, average=None)
    auc_avg = MulticlassAUROC(num_classes=args.num_classes, average="macro")
    pre = MulticlassPrecision(num_classes=args.num_classes, average=None)
    pre_avg = MulticlassPrecision(num_classes=args.num_classes, average="micro")
    f1 = MulticlassF1Score(num_classes=args.num_classes, average=None)
    f1_avg = MulticlassF1Score(num_classes=args.num_classes, average="micro")

    _, _, _, _, test_features, test_labels = data_loader(data, args.num_samples_stu)
    student = torch.load("../result/vanilla_kd_student.pkl")     # load student network
    _, test_pred = student(torch.from_numpy(test_features).float())
    print("Testing accuracy in each class:{}".format(acc(test_pred, torch.from_numpy(test_labels))))
    print("Average testing accuracy:{}".format(acc_avg(test_pred, torch.from_numpy(test_labels))))
    print("Testing AUC score in each class:{}".format(auc(test_pred, torch.from_numpy(test_labels))))
    print("Average testing AUC score:{}".format(auc_avg(test_pred, torch.from_numpy(test_labels))))
    print("Testing cross entropy loss:{}".format(loss_func(test_pred, torch.from_numpy(test_labels))))
    print("Testing precision in each class:{}".format(pre(test_pred, torch.from_numpy(test_labels))))
    print("Average testing precision:{}".format(pre_avg(test_pred, torch.from_numpy(test_labels))))
    print("Testing F1 score in each class:{}".format(f1(test_pred, torch.from_numpy(test_labels))))
    print("Average testing F1 score in each class:{}".format(f1_avg(test_pred, torch.from_numpy(test_labels))))



if __name__ == '__main__':
    dataset = pd.read_csv('../data/data.csv')
    minMax = MinMaxScaler()
    sampler = RandomOverSampler()
    args = parse_args()
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    loss_func = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss()

    # vanilla KD: train teacher and student network
    # print("Teacher training start.")
    # teacher = teacher_train(dataset)
    # teacher_test(dataset)
    # print("Student training start.")
    student = student_train(dataset)
    # student_test(dataset)