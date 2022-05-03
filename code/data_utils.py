import string

import numpy as np
import pandas as pd
from sklearn import model_selection
import torch
from torch.utils.data import TensorDataset

import utils


def categorical_encoding(df):
    dropped = ["f_27"]
    for letter in string.ascii_letters:
            df[letter] = df["f_27"].str.count(letter)
            if 0 == df[letter].sum():
                dropped.append(letter)

    return df.drop(dropped, axis=1)


def data_pipeline(df, train=True):
    df = categorical_encoding(df)

    if train:
        X = df.drop("target", axis=1).values
        y = df["target"].values
    else:
        X, y = df.values, None

    del df

    return X, y
