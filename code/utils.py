from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn import model_selection, metrics
from tqdm import tqdm


TRAINPATH = Path("../input/train.csv")
TESTPATH = Path("../input/test.csv")
SUBMISSIONPATH = Path("../input/sample_submission.csv")


class CrossValidator:
    def __init__(self, X, y, n_splits=5, seed=None):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.seed = seed
        self.skf = model_selection.StratifiedKFold(
            n_splits=n_splits, random_state=seed)

    def __iter__(self):
        for train_idx, dev_idx in self.skf.split(self.X, self.y):
            X_train, X_dev = self.X[train_idx], self.X[dev_idx]
            y_train, y_dev = self.y[train_idx], self.y[dev_idx]

            traindata = TensorDataset(
                torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train))
            devdata = TensorDataset(
                torch.tensor(X_dev, dtype=torch.float), torch.tensor(y_dev))

            yield traindata, devdata


def training_data(**kwargs):
    return pd.read_csv(TRAINPATH, index_col=0, **kwargs)


def test_data(**kwargs):
    return pd.read_csv(TESTPATH, index_col=0, **kwargs)


def accuracy_score_logits(logits, true, normalize=False):
    score = torch.sum(true == logits.argmax(dim=1)).item()

    return score / len(true) if normalize else score


# def roc_auc_score(logits, true):
#     preds = logits[:, 1].detach().tolist()
#     true = true.detach().tolist()
#     score = metrics.roc_auc_score(true, preds)

#     return score


def register_predictions(predictions, filename="submission.csv"):
    preds = pd.read_csv(SUBMISSIONPATH, index_col="id")
    preds["target"] = predictions

    preds.to_csv(filename)


def save_checkpoint(model_state,
                    optim_state,
                    file_name,
                    **params):
    """Checkpoint model params during training."""
    checkpoint = {
        "model_state_dict": model_state,
        "optim_state_dict": optim_state
    }
    for key, val in params.items():
        checkpoint[key] = val

    torch.save(checkpoint, file_name)


def load_checkpoint(file_name):
    """Retrieve saved model state dict."""
    return torch.load(file_name)


def train_batch(net,
                dataloader,
                optimizer,
                criterion,
                device=None,
                quiet=False):
    """pytorch training loop."""
    net.train()
    acc = total_loss = roc_score = 0
    y_true, y_score = [], []

    for X, y in tqdm(dataloader, leave=False, disable=quiet):
        if device:
            X, y = X.to(device), y.to(device)
        logits = net(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_score.extend(logits.detach()[:,1].tolist())
        y_true.extend(y.detach().tolist())

        total_loss += loss.item()
        acc += accuracy_score_logits(logits, y, normalize=False)

    acc /= len(dataloader.dataset)
    roc_score = metrics.roc_auc_score(y_true, y_score)

    return total_loss, acc, roc_score


@torch.no_grad()
def validate(net,
             dataloader,
             objective,
             device=None,
             quiet=False):
    """pytorch inference loop."""
    net.eval()
    acc = total_loss = 0
    y_true, y_score = [], []

    for X, y in tqdm(dataloader, leave=False, disable=quiet):
        if device:
            X, y = X.to(device), y.to(device)
        logits = net(X)

        y_score.extend(logits.detach()[:,1].tolist())
        y_true.extend(y.detach().tolist())

        acc += accuracy_score_logits(logits, y, normalize=False)
        total_loss += objective(logits, y).item()

    acc /= len(dataloader.dataset)
    roc_score = metrics.roc_auc_score(y_true, y_score)

    return total_loss, acc, roc_score


def inference(net, dataloader):
    """Use trained model for inference."""
    net.eval()
    with torch.no_grad():
        predictions = []
        for (inp,) in dataloader:
            out = net(inp).detach()[:,1]
            predictions.extend(out.tolist())

    return predictions
