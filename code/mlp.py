import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
import wandb

import data_preprocess
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
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        out = self.layers(x)
        if self.return_logits is False:
            out = torch.softmax(out, dim=1)

        return out


def train_batch(net,
                dataloader,
                optimizer,
                criterion,
                device=None,
                quiet=False):
    """pytorch training loop."""
    net.train()
    acc = total_loss = 0
    for X, y in tqdm(dataloader, leave=False, disable=quiet):
        if device:
            X, y = X.to(device), y.to(device)
        logits = net(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        acc += utils.accuracy_score_logits(logits, y, normalize=False)

    return total_loss, (acc / len(dataloader.dataset))


@torch.no_grad()
def validate(net,
             dataloader,
             objective,
             device=None,
             quiet=False):
    """pytorch inference loop."""
    net.eval()
    score = total_loss = 0
    for X, y in tqdm(dataloader, leave=False, disable=quiet):
        if device:
            X, y = X.to(device), y.to(device)
        logits = net(X)
        score += utils.accuracy_score_logits(logits, y, normalize=False)
        total_loss += objective(logits, y).item()

    return total_loss, score / len(dataloader.dataset)


def train(config):
    train_df = utils.training_data()
    X, y = data_preprocess.data_pipeline(train_df)
    cv = utils.CrossValidator(X, y, n_splits=config.n_splits)

    for split, (traindata, devdata) in enumerate(cv):
        print(f"split={split}")
        net = MLP(in_features=X.shape[1], out_features=2, return_logits=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()

        wandb.watch(net)

        trainloader = DataLoader(
            traindata, shuffle=True, batch_size=config.batch_size)
        devloader = DataLoader(
            devdata, shuffle=True, batch_size=config.batch_size)

        for epoch in trange(config.n_epochs):
            trainloss, trainacc = train_batch(
                net, trainloader, optimizer, criterion)
            devloss, devacc = validate(
                net, devloader, criterion)

            wandb.log({
                f"split{split}/train/loss": trainloss,
                f"split{split}/train/acc": trainacc,
                f"split{split}/dev/loss": devloss,
                f"split{split}/dev/acc": devacc,
                f"split{split}/epoch": epoch
            })

            print(f"{epoch=}")
            print(f"\t{trainloss=:0.2f}, {trainacc=:0.2f}")
            print(f"\t{devloss=:0.2f}, {devacc=:0.2f}")


def main(config):
    train_df = utils.training_data()
    test_df = utils.test_data()
    X_train, y_train = data_preprocess.data_pipeline(train_df, train=True)
    X_test, _ = data_preprocess.data_pipeline(test_df, train=False)

    traindata = TensorDataset(torch.tensor(X_train, dtype=torch.float),
                              torch.tensor(y_train))
    testdata = TensorDataset(torch.tensor(X_test, dtype=torch.float))

    net = MLP(in_features=X_train.shape[1], out_features=2, return_logits=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    wandb.watch(net)

    trainloader = DataLoader(
        traindata, shuffle=True, batch_size=config.batch_size)
    testloader = DataLoader(
        testdata, shuffle=False, batch_size=len(testdata))

    for epoch in trange(config.n_epochs):
        trainloss, trainacc = train_batch(
            net, trainloader, optimizer, criterion)

        wandb.log({
            f"train/loss": trainloss,
            f"train/acc": trainacc,
            f"epoch": epoch
        })

        print(f"{epoch=}")
        print(f"\t{trainloss=:0.2f}, {trainacc=:0.2f}")

    utils.save_checkpoint(net.state_dict(),
                          optimizer.state_dict(),
                          file_name=f"logs/mlp-{wandb.run.name}.pt")


if __name__ == "__main__":
    config = dict(batch_size=32,
                  lr=1e-3,
                  n_splits=5,
                  n_epochs=10)

    with wandb.init(project="kaggle-tabular", config=config) as run:
        config = wandb.config
        main(config)
