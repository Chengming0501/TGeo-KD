import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC
from prettytable import PrettyTable

# binary_accuracy_metric = BinaryAccuracy()
accuracy = Accuracy(num_classes=2, task='multiclass')
auroc = AUROC(num_classes=2, average='macro', task='multiclass')

# Apply the softmax function to the logits
softmax = nn.Softmax(dim=1)

# Compute the NLL metric
nll_loss = nn.NLLLoss()
from sklearn.metrics import confusion_matrix


def compute_metrics(model, dataloader, device):
    predicts, GT_labels = [], []
    for x, y_g in dataloader:
        x, y_g = x.to(device).float(), y_g.to(device).long()
        y_prediction = model(x)
        predicts.extend(y_prediction.tolist())
        GT_labels.extend(y_g.tolist())

    predicts = torch.tensor(predicts)
    GT_labels = torch.tensor(GT_labels)

    # compute acc, auc, nll
    acc = accuracy(predicts, GT_labels).item()
    auc = auroc(softmax(predicts), GT_labels).item()
    nll = nll_loss(F.log_softmax(predicts, dim=1), GT_labels).item()

    # compute the confusion matrix
    preds = torch.argmax(predicts, dim=1).numpy()
    labels = GT_labels.numpy()
    cm = confusion_matrix(labels, preds)
    # print(cm)

    return acc, auc, nll
