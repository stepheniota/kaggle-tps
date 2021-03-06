from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
import wandb

import data_utils
import utils


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=2,
                 return_logits=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.return_logits = return_logits

        self.layers = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        out = self.layers(x)
        if self.return_logits is False:
            out = torch.softmax(out, dim=1)

        return out


def cross_validate(config_):
    train_df = data_utils.training_data()
    X, y = data_utils.data_pipeline(train_df)
    cv = data_utils.CrossValidator(X, y, n_splits=config_.n_splits)

    for fold, (traindata, devdata) in enumerate(cv):
        run = wandb.init(project="may2022-tabular-kaggle",
                         group="testing_cv",
                         config=config_._asdict(),
                         notes=f"Cross validation fold {fold}.")

        config = wandb.config

        print(f"\tfold={fold}")
        net = MLP(in_features=X.shape[1], out_features=2, return_logits=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()

        trainloader = DataLoader(
            traindata, shuffle=True, batch_size=config.batch_size)
        devloader = DataLoader(
            devdata, shuffle=True, batch_size=config.batch_size)

        # train_losses = dev_losses = train_accs = dev_accs = 0.
        # train_roc_scores = dev_roc_scores = 0.

        for epoch in range(config.n_epochs):
            trainloss, trainacc, trainroc = utils.train_batch(
                net, trainloader, optimizer, criterion)
            devloss, devacc, devroc = utils.validate(
                net, devloader, criterion)

            wandb.log(
                {f"train/loss": trainloss,
                 f"train/acc": trainacc,
                 f"train/roc_auc": trainroc,
                 f"dev/loss": devloss,
                 f"dev/acc": devacc,
                 f"dev/roc_auc": devroc,},
            )

            print(f"\t{epoch=}")
            print(f"\t\t{trainloss=:0.2f}, {trainacc=:0.2f}, {trainroc=:0.2f}")
            print(f"\t\t{devloss=:0.2f}, {devacc=:0.2f}, {devroc=:0.2f}")

        #     train_losses += trainloss
        #     dev_losses += devloss
        #     train_accs += trainacc
        #     dev_accs += devacc
        #     train_roc_scores += trainroc
        #     dev_roc_scores += devroc

        # wandb.log(
        #     {"ave/train/loss": train_losses/config.n_splits,
        #      "ave/dev/loss": dev_losses / config.n_splits,
        #      "ave/train/accs": train_accs / config.n_splits,
        #      "ave/dev/accs": dev_accs / config.n_splits,
        #      "ave/train/roc_auc": train_roc_scores / config.n_splits,
        #      "ave/dev/roc_auc": dev_roc_scores / config.n_splits,},
        # )

        if 2 == fold:
            return

def train(config):
    run = wandb.init(project="may2022-tabular-kaggle", config=config._asdict())
    config = wandb.config

    train_df = data_utils.training_data()
    X_train, y_train = data_utils.data_pipeline(train_df, train=True)

    traindata = TensorDataset(torch.tensor(X_train, dtype=torch.float),
                              torch.tensor(y_train))

    net = MLP(in_features=X_train.shape[1], out_features=2, return_logits=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    trainloader = DataLoader(
        traindata, shuffle=True, batch_size=config.batch_size)

    for epoch in trange(config.n_epochs):
        trainloss, trainacc, trainroc = utils.train_batch(
            net, trainloader, optimizer, criterion)

        wandb.log({"train/loss": trainloss,
                   "train/acc": trainacc,
                   "train/roc_auc": trainroc,},)

        print(f"\t{epoch=}")
        print(f"\t\t{trainloss=:0.2f}, {trainacc=:0.2f}")

    utils.save_checkpoint(net.state_dict(),
                          optimizer.state_dict(),
                          file_name=f"logs/mlp-{run.name}.pt")

    wandb.finish()

def predict(statedict_path):
    test_df = data_utils.test_data()
    X_test, _ = data_utils.data_pipeline(test_df, train=False)
    testdata = TensorDataset(torch.tensor(X_test, dtype=torch.float))
    testloader = DataLoader(testdata, shuffle=False, batch_size=64)

    model_state_dict = utils.load_checkpoint(
        file_name=statedict_path)["model_state_dict"]
    net = MLP(in_features=X_test.shape[1], out_features=2)
    net.load_state_dict(model_state_dict)
    net.return_logits = False

    predictions = utils.inference(net, testloader)
    utils.register_predictions(predictions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train MLP on tabular data.")
    parser.add_argument(
        "--mode", "-m", type=str, choices=("test", "train", "cv"))
    parser.add_argument("--checkpoint", "-c", type=str, required=False)
    args = parser.parse_args()

    config = dict(batch_size=32, lr=3e-4, n_splits=5, n_epochs=11)
    config = namedtuple("Params", config.keys())(**config)


    if args.mode == "test":
        assert args.checkpoint is not None
        predict(args.checkpoint)
    elif args.mode == "cv":
        cross_validate(config)
    elif args.mode == "train":
        train(config)
    else:
        raise ValueError
