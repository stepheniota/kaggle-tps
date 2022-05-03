import string

import numpy as np
import pandas as pd
from sklearn import model_selection
import torch
from torch.utils.data import TensorDataset

import utils


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
    return pd.read_csv(utils.TRAINPATH, index_col=0, **kwargs)


def test_data(**kwargs):
    return pd.read_csv(utils.TESTPATH, index_col=0, **kwargs)


def char2count_encoding(df):
    dropped = ["f_27"]
    for letter in string.ascii_letters:
            df[letter] = df["f_27"].str.count(letter)
            if 0 == df[letter].sum():
                dropped.append(letter)

    return df.drop(dropped, axis=1)


def positional_encoding(df):
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')

    return df.drop(["f_27"], axis=1)


def data_pipeline(df, train=True):
    df = char2count_encoding(df)

    if train:
        X = df.drop("target", axis=1).values
        y = df["target"].values
    else:
        X, y = df.values, None

    del df

    return X, y
