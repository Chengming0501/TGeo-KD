import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.01)


class Controller(nn.Module):
    def __init__(self, input_dim):
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=32, bias=True)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(in_features=16, out_features=1, bias=True)
        # self.softmax = F.sigmoid()

    def forward(self, input_data):
        output = self.fc1(input_data)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        output = F.sigmoid(output)
        return output



class Student(nn.Module):
    def __init__(self, input_dim):
        super(Student, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=64)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(in_features=32, out_features=3)
        self.softmax = nn.Softmax()

    def forward(self, input_data):
        output = self.fc1(input_data)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        knowledge = output
        output = self.softmax(output)
        return knowledge, output


class Teacher(nn.Module):
    def __init__(self, input_dim):
        super(Teacher, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=512)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.relu4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(in_features=64, out_features=32)
        self.relu5 = nn.LeakyReLU()
        self.fc6 = nn.Linear(in_features=32, out_features=3)
        self.softmax = nn.Softmax()

    def forward(self, input_data):
        output = self.fc1(input_data)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        output = self.relu3(output)
        output = self.fc4(output)
        output = self.relu4(output)
        output = self.fc5(output)
        output = self.relu5(output)
        output = self.fc6(output)
        knowledge = output
        output = self.softmax(output)
        return knowledge, output


#
# def teacher_train(target_cell_name, trunk_length, output_length):
#     train_test_ratio = 0.8
#     epochs = 200
#     x_train, y_train, x_test, y_test = readtxt.read_data_per_cell_seq(cellID=target_cell_name,
#                                                                       train_test_ratio=train_test_ratio,
#                                                                       trunk_length=trunk_length,
#                                                                       output_length=output_length)
#     [m, n] = np.shape(x_train)
#     teacher = Teacher(input_dim=n, output_len=output_length)
#     optimizer = torch.optim.Adam(teacher.parameters(), lr=0.01)
#     loss_func = torch.nn.MSELoss()
#     x = torch.from_numpy(x_train)
#     y = torch.from_numpy(y_train)
#     # Train the teacher
#     for ep in range(epochs):
#         pred = teacher(x.float())
#         loss = loss_func(pred, y.float())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print("The " + str(ep) + "-th epoch loss: " + str(loss))
#     return teacher
#
# def student_train(target_cell_name, trunk_length, output_length):
#     train_test_ratio = 0.8
#     epochs = 200
#     x_train, y_train, x_test, y_test = readtxt.read_data_per_cell_seq(cellID=target_cell_name,
#                                                                       train_test_ratio=train_test_ratio,
#                                                                       trunk_length=trunk_length,
#                                                                       output_length=output_length)
#     [m, n] = np.shape(x_train)
#     student = Student(input_dim=n, output_len=output_length)
#     optimizer = torch.optim.Adam(student.parameters(), lr=0.01)
#     loss_func = torch.nn.MSELoss()
#     x = torch.from_numpy(x_train)
#     y = torch.from_numpy(y_train)
#     # Train the student
#     for ep in range(epochs):
#         pred = student(x.float())
#         loss = loss_func(pred, y.float())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print("The " + str(ep) + "-th epoch loss: " + str(loss))
#     return student
