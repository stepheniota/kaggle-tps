import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
import wandb

import data_preprocess
import utils
import mlp


def main(statedict_path):
    test_df = utils.test_data()
    X_test, _ = data_preprocess.data_pipeline(test_df, train=False)
    testdata = TensorDataset(torch.tensor(X_test, dtype=torch.float))
    testloader = DataLoader(testdata, shuffle=False, batch_size=64)

    model_state_dict = utils.load_checkpoint(
        file_name=statedict_path)["model_state_dict"]
    net = mlp.MLP(in_features=X_test.shape[1], out_features=2, return_logits=True)
    net.load_state_dict(model_state_dict)
    net.return_logits = False

    net.eval()
    with torch.no_grad():
        predictions = []
        for (inp,) in testloader:
            probabilites = net(inp).detach()[:,1]
            predictions.extend(probabilites.tolist())

    utils.register_predictions(predictions)


if __name__ == "__main__":
    main(sys.argv[1])
