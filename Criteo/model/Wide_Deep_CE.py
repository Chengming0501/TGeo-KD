import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, dropout_rate, ):
        super(DNN, self).__init__()
        self.inputs_dim = inputs_dim
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_units = [inputs_dim] + list(self.hidden_units)
        self.linear = nn.ModuleList([
            nn.Linear(self.hidden_units[i], self.hidden_units[i + 1]) for i in range(len(self.hidden_units) - 1)
        ])
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(tensor)

        self.activation = nn.ReLU()

    def forward(self, X):
        inputs = X
        for i in range(len(self.linear)):
            fc = self.linear[i](inputs)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            inputs = fc
        return inputs


class Wide_Deep(nn.Module):
    def __init__(self, feat_size, embedding_size, linear_feature_columns, dnn_feature_columns, dnn_hidden_units,
                 drop_rate=0.9, use_attention=True, attention_factor=8, l2_reg=0.00001):
        super(Wide_Deep, self).__init__()
        self.sparse_feature_columns = list(filter(lambda x: x[1] == 'sparse', dnn_feature_columns))
        self.embedding_dic = nn.ModuleDict({
            feat[0]: nn.Embedding(feat_size[feat[0]], embedding_size, sparse=False) for feat in
            self.sparse_feature_columns
        })
        self.dense_feature_columns = list(filter(lambda x: x[1] == 'dense', dnn_feature_columns))

        self.feature_index = defaultdict(int)
        start = 0
        for feat in feat_size:
            self.feature_index[feat] = start
            start += 1

        self.dnn = DNN(len(self.dense_feature_columns) + embedding_size * len(self.embedding_dic), dnn_hidden_units,
                       0.5)

        # self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 2, bias=False)

        # dnn_hidden_units = [len(feat_size), 1]
        dnn_hidden_units = [len(feat_size), 2]
        self.linear = nn.ModuleList([
            nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i + 1]) for i in range(len(dnn_hidden_units) - 1)
        ])
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(tensor)

        # self.out = nn.Sigmoid()
        self.out = nn.Softmax(dim=1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):

        # wide
        logit = X
        for i in range(len(self.linear)):
            fc = self.linear[i](logit)
            fc = self.act(fc)
            fc = self.dropout(fc)
            logit = fc

        # deep
        sparse_embedding = [
            self.embedding_dic[feat[0]](X[:, self.feature_index[feat[0]]].long()).reshape(X.shape[0], 1, -1)
            for feat in self.sparse_feature_columns]
        sparse_input = torch.cat(sparse_embedding, dim=1)
        sparse_input = torch.flatten(sparse_input, start_dim=1)
        dense_values = [X[:, self.feature_index[feat[0]]].reshape(-1, 1) for feat in self.dense_feature_columns]
        dense_input = torch.cat(dense_values, dim=1)
        dnn_input = torch.cat((sparse_input, dense_input), dim=1)
        dnn_out = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_out)

        logit += dnn_logit
        # print("dnn_input:", dnn_input.shape)
        # print("dnn_out:", dnn_out.shape)
        # print("dnn_logit:", dnn_logit.shape)
        # print("logit:", logit.shape)

        # y_pred = torch.sigmoid(logit)
        # logit = self.out(logit)
        return logit


class Controller(nn.Module):
    def __init__(self, dim1, device):
        super(Controller, self).__init__()

        self.linear1 = nn.Linear(dim1, 8, bias=True).to(device)
        self.linear2 = nn.Linear(8, 1, bias=False).to(device)

    def forward(self, x):
        z1 = torch.relu(self.linear1(x))
        res = torch.sigmoid(self.linear2(z1))

        return res
