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
            # nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        out = self.layers(x)
        if self.return_logits is False:
            out = torch.softmax(out, dim=1)

        return out


def cross_validate(config):
    train_df = utils.training_data()
    X, y = data_utils.data_pipeline(train_df)
    cv = utils.CrossValidator(X, y, n_splits=config.n_splits)

    for fold, (traindata, devdata) in enumerate(cv):
        print(f"fold={fold}")
        net = MLP(in_features=X.shape[1], out_features=2, return_logits=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()

        wandb.watch(net)

        trainloader = DataLoader(
            traindata, shuffle=True, batch_size=config.batch_size)
        devloader = DataLoader(
            devdata, shuffle=True, batch_size=config.batch_size)

        for epoch in trange(config.n_epochs):
            trainloss, trainacc, trainroc = utils.train_batch(
                net, trainloader, optimizer, criterion)
            devloss, devacc, devroc = utils.validate(
                net, devloader, criterion)

            wandb.log({
                f"fold{fold}/train/loss": trainloss,
                f"fold{fold}/train/acc": trainacc,
                f"fold{fold}/train/roc_auc": trainroc,
                f"fold{fold}/dev/loss": devloss,
                f"fold{fold}/dev/acc": devacc,
                f"fold{fold}/dev/roc_auc": devroc,
                f"fold{fold}/epoch": epoch
            })

            print(f"{epoch=}")
            print(f"\t{trainloss=:0.2f}, {trainacc=:0.2f}, {trainroc=:0.2f}")
            print(f"\t{devloss=:0.2f}, {devacc=:0.2f}, {devroc=:0.2f}")


def train(config):
    train_df = utils.training_data()
    # test_df = utils.test_data()
    X_train, y_train = data_utils.data_pipeline(train_df, train=True)
    # X_test, _ = data_utils.data_pipeline(test_df, train=False)

    traindata = TensorDataset(torch.tensor(X_train, dtype=torch.float),
                              torch.tensor(y_train))
    # testdata = TensorDataset(torch.tensor(X_test, dtype=torch.float))

    net = MLP(in_features=X_train.shape[1], out_features=2, return_logits=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    wandb.watch(net)

    trainloader = DataLoader(
        traindata, shuffle=True, batch_size=config.batch_size)
    # testloader = DataLoader(
    #     testdata, shuffle=False, batch_size=len(testdata))

    for epoch in trange(config.n_epochs):
        trainloss, trainacc, trainroc = utils.train_batch(
            net, trainloader, optimizer, criterion)

        wandb.log({
            f"train/loss": trainloss,
            f"train/acc": trainacc,
            f"train/roc_auc": trainroc,
            f"epoch": epoch
        })

        print(f"{epoch=}")
        print(f"\t{trainloss=:0.2f}, {trainacc=:0.2f}")

    utils.save_checkpoint(net.state_dict(),
                          optimizer.state_dict(),
                          file_name=f"logs/mlp-{wandb.run.name}.pt")

def predict(statedict_path):
    test_df = utils.test_data()
    X_test, _ = data_utils.data_pipeline(test_df, train=False)
    testdata = TensorDataset(torch.tensor(X_test, dtype=torch.float))
    testloader = DataLoader(testdata, shuffle=False, batch_size=64)

    model_state_dict = utils.load_checkpoint(
        file_name=statedict_path)["model_state_dict"]
    net = MLP(in_features=X_test.shape[1], out_features=2)
    net.load_state_dict(model_state_dict)
    net.return_logits = False

    predictions = utils.inference(net, testloader)
    # net.eval()
    # with torch.no_grad():
    #     predictions = []
    #     for (inp,) in testloader:
    #         probabilites = net(inp).detach()[:,1]
    #         predictions.extend(probabilites.tolist())

    utils.register_predictions(predictions)


if __name__ == "__main__":
    import sys; argc = len(sys.argv); mode = sys.argv[1]
    assert argc > 1 and mode in ("cv", "train", "test")
    if mode == "test": assert argc == 3

    config = dict(batch_size=32,
                  lr=3e-4,
                  n_splits=5,
                  n_epochs=10)

    if mode == "test":
        predict(sys.argv[2])
    elif mode == "cv" or mode == "train":
        with wandb.init(project="may2022-tabular-kaggle", config=config):
            config = wandb.config
            if mode == "cv":
                cross_validate(config)
            else:
                train(config)
    else:
        raise ValueError
