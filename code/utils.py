from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn import model_selection


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
