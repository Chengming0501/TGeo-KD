import os
import torch
from network import Student, Teacher, Controller
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from collections import Counter
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch.nn.functional as F


def parse_args():
    parser = ArgumentParser(description="control_kd")
    parser.add_argument('--num_samples_tea', type=int, default=50000, help="numbers of sample for training teacher")
    parser.add_argument('--num_samples_stu', type=int, default=10000, help="numbers of sample for training student")
    parser.add_argument('--seed', type=int, default=0, help="random state for spitting data (for reproducability)")
    parser.add_argument('--temperature', type=float, default=1.3, help="temperature")
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate for student and teacher training")
    parser.add_argument('--num_epochs_tea', type=int, default=500, help="number of epoch for training teacher")
    parser.add_argument('--num_epochs_stu', type=int, default=500, help="number of epoch for training student")
    parser.add_argument('--check', type=int, default=10, help="epoch of check point")
    parser.add_argument('--early_stop', type=float, default=1e-3, help="early stopping criteria")
    parser.add_argument('--dim_option_1', type=int, default=12, help="option 1: input with probability")
    parser.add_argument('--dim_option_2', type=int, default=9, help="option 2: input with distance of probability")
    parser.add_argument('--dim_option_3', type=int, default=21, help="option 3: input with distance and probability")
    parser.add_argument('--dim_option_4', type=int, default=18, help="option 4: input with angle and probability")
    parser.add_argument('--dim_option_5', type=int, default=15, help="option 5: input with angle and distance")
    parser.add_argument('--dim_option_6', type=int, default=27, help="option 6: input with angle, distance and probability")

    return parser.parse_args()



def data_loader(data, num_samples):
    labels = data.iloc[:, -1].to_numpy()
    features = data.iloc[:, :-1].to_numpy()
    features = minMax.fit_transform(features)  # normalization
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
        teacher.train()
        _, train_pred = teacher(torch.from_numpy(train_features).float())
        train_loss = loss_func(train_pred, torch.from_numpy(train_labels))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print("The " + str(ep) + "-th epoch training loss: " + str(train_loss))

        teacher.eval()
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

    torch.save(teacher, "../result/teacher.pkl")      # save the teacher network

    return teacher


def teacher_test(data):
    acc = MulticlassAccuracy(num_classes=args.num_classes, average=None)
    acc_avg = MulticlassAccuracy(num_classes=args.num_classes, average="micro")
    auc = MulticlassAUROC(num_classes=args.num_classes, average=None)
    auc_avg = MulticlassAUROC(num_classes=args.num_classes, average="macro")

    _, _, _, _, test_features, test_labels = data_loader(data, args.num_samples_stu)
    teacher = torch.load("../result/teacher.pkl")        # load teacher network
    _, test_pred = teacher(torch.from_numpy(test_features).float())
    print("Testing accuracy in each class:{}".format(acc(test_pred, torch.from_numpy(test_labels))))
    print("Average testing accuracy:{}".format(acc_avg(test_pred, torch.from_numpy(test_labels))))
    print("Testing AUC score in each class:{}".format(auc(test_pred, torch.from_numpy(test_labels))))
    print("Average testing AUC score:{}".format(auc_avg(test_pred, torch.from_numpy(test_labels))))
    print("Testing cross entropy loss:{}".format(torch.mean(loss_func(test_pred, torch.from_numpy(test_labels)))))



def state_design(student_pred_prob, teacher_pred_prob, gt_prob):
    # average teacher predicted probability in each class
    [m, n] = gt_prob.size()
    avg_teacher_pred_prob = torch.zeros(m, n)
    for i in range(args.num_classes):
        sample_index = np.where(gt_prob[:, i] == 1)
        avg_teacher_pred_prob[sample_index, :] = torch.mean(teacher_pred_prob[sample_index, :], dim=1)

    # option 1: predicted probability of student, teacher, average teacher, and ground truth
    # mlp_state = torch.cat((student_pred_prob, teacher_pred_prob, gt_prob, avg_teacher_pred_prob), dim=1)

    # option 2: distance between the predicted prob of student, teacher, and ground truth
    # dis_os = torch.square(avg_teacher_pred_prob - student_pred_prob)
    # dis_gs = torch.square(gt_prob - student_pred_prob)
    # dis_ts = torch.square(teacher_pred_prob - student_pred_prob)
    # mlp_state = torch.cat((dis_os, dis_gs, dis_ts), dim=1)

    # option 3: distance and predicted probability
    # dis_os = torch.square(avg_teacher_pred_prob - student_pred_prob)
    # dis_gs = torch.square(gt_prob - student_pred_prob)
    # dis_ts = torch.square(teacher_pred_prob - student_pred_prob)
    # mlp_state = torch.cat((student_pred_prob, teacher_pred_prob, gt_prob, avg_teacher_pred_prob, dis_os, dis_gs, dis_ts), dim=1)

    # option 4: angle and predicted probability
    # ang_osg = (avg_teacher_pred_prob - student_pred_prob) * (gt_prob - student_pred_prob)
    # ang_tsg = (teacher_pred_prob - student_pred_prob) * (gt_prob - student_pred_prob)
    # mlp_state = torch.cat((student_pred_prob, teacher_pred_prob, gt_prob, avg_teacher_pred_prob, ang_osg, ang_tsg), dim=1)

    # option 5: distance and angle
    # dis_os = torch.square(avg_teacher_pred_prob - student_pred_prob)
    # dis_gs = torch.square(gt_prob - student_pred_prob)
    # dis_ts = torch.square(teacher_pred_prob - student_pred_prob)
    # ang_osg = (avg_teacher_pred_prob - student_pred_prob) * (gt_prob - student_pred_prob)
    # ang_tsg = (teacher_pred_prob - student_pred_prob) * (gt_prob - student_pred_prob)
    # mlp_state = torch.cat((dis_os, dis_gs, dis_ts, ang_osg, ang_tsg), dim=1)

    # option 6: distance, angle and predicted probability
    dis_os = torch.square(avg_teacher_pred_prob - student_pred_prob)
    dis_gs = torch.square(gt_prob - student_pred_prob)
    dis_ts = torch.square(teacher_pred_prob - student_pred_prob)
    ang_osg = (avg_teacher_pred_prob - student_pred_prob) * (gt_prob - student_pred_prob)
    ang_tsg = (teacher_pred_prob - student_pred_prob) * (gt_prob - student_pred_prob)
    mlp_state = torch.cat((student_pred_prob, teacher_pred_prob, gt_prob, avg_teacher_pred_prob,
                           dis_os, dis_gs, dis_ts, ang_osg, ang_tsg), dim=1)

    return mlp_state



def student_train(data):

    train_features, train_labels, valid_features, valid_labels, test_features, test_labels = data_loader(data, args.num_samples_stu)
    [m, n] = np.shape(train_features)
    gt_prob = np.eye(args.num_classes)[train_labels]
    gt_prob = torch.from_numpy(gt_prob)
    teacher = torch.load("../result/teacher.pkl")   # load teacher network
    student = Student(input_dim=n)       # student network
    controller = Controller(input_dim=args.dim_option_6)        # tradeoff weight MLP
    optimizer_student = torch.optim.Adam(student.parameters(), lr=0.01)
    optimizer_controller = torch.optim.Adam(controller.parameters(), lr=0.01, weight_decay=0.5)
    valid_loss_list = []

    # Train the student network
    for ep in range(args.num_epochs_stu):
        student.train()
        student_knowledge, student_pred_prob = student(torch.from_numpy(train_features).float())
        teacher_knowledge, teacher_pred_prob = teacher(torch.from_numpy(train_features).float())
        student_knowledge_soft = nn.functional.log_softmax(student_knowledge / args.temperature, dim=1)
        teacher_knowledge_soft = nn.functional.softmax(teacher_knowledge / args.temperature, dim=1)
        kd_loss = kl_loss(student_knowledge_soft, teacher_knowledge_soft).sum(1)      # knowledge distillation loss
        gt_loss = loss_func(student_pred_prob, torch.from_numpy(train_labels))     # ground truth loss
        mlp_state = state_design(student_pred_prob, teacher_pred_prob, gt_prob)    # mlp input design
        alpha = torch.squeeze(controller(mlp_state.float()))       # learned tradeoff weight
        loss = alpha * args.temperature ** 2 * kd_loss + (1 - alpha) * gt_loss     # total loss
        loss = torch.mean(loss)
        optimizer_student.zero_grad()
        optimizer_controller.zero_grad()
        loss.backward()
        optimizer_student.step()
        optimizer_controller.step()
        print("The " + str(ep) + "-th epoch training loss: " + str(loss))
        print("The tradeoff weight: ", torch.max(alpha), torch.mean(alpha), torch.min(alpha))

        student.eval()
        with torch.no_grad():
            _, valid_pred = student(torch.from_numpy(valid_features).float())
            valid_loss = torch.mean(loss_func(valid_pred, torch.from_numpy(valid_labels)))
            print("The " + str(ep) + "-th epoch validation loss: " + str(valid_loss))
            valid_loss_list.append(valid_loss)
            # check early stopping criteria
            if ep >= args.check:
                valid_loss_check = np.array(valid_loss_list[ep - args.check: ep])
                if all(i <= args.early_stop for i in np.absolute(np.diff(valid_loss_check))):
                    print("Early stopping")
                    break

    torch.save(student, "../result/control_kd_student_option_6.pkl")  # save the student network

    return student


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
    student = torch.load("../result/control_kd_student_option_6.pkl")     # load student network
    _, test_pred = student(torch.from_numpy(test_features).float())
    print("Testing accuracy in each class:{}".format(acc(test_pred, torch.from_numpy(test_labels))))
    print("Average testing accuracy:{}".format(acc_avg(test_pred, torch.from_numpy(test_labels))))
    print("Testing AUC score in each class:{}".format(auc(test_pred, torch.from_numpy(test_labels))))
    print("Average testing AUC score:{}".format(auc_avg(test_pred, torch.from_numpy(test_labels))))
    print("Testing cross entropy loss:{}".format(torch.mean(loss_func(test_pred, torch.from_numpy(test_labels)))))
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
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    kl_loss = torch.nn.KLDivLoss(reduction='none')

    # train teacher and student network
    # print("Teacher training start.")
    # teacher = teacher_train(dataset)
    # print("Student training start.")
    student = student_train(dataset)
    # student_test(dataset)
