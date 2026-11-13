"""Funções da entrega de classificação do projeto."""

import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_values(y_true, predictions, probabilities=None):
    values = {
        "accuracy": accuracy_score(y_true, predictions),
        "f1_macro": f1_score(y_true, predictions, average="macro"),
    }
    if probabilities is not None:
        values["roc_auc_ovr_macro"] = roc_auc_score(
            y_true, probabilities, multi_class="ovr", average="macro"
        )
    return values


def fit_baselines(x_train, y_train, x_val, y_val, seed=42):
    dummy = DummyClassifier(strategy="most_frequent", random_state=seed)
    linear = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed
    )
    dummy.fit(x_train, y_train)
    linear.fit(x_train, y_train)
    table = pd.DataFrame([
        {"modelo": "Classe majoritária", **metric_values(y_val, dummy.predict(x_val))},
        {"modelo": "Regressão logística", **metric_values(y_val, linear.predict(x_val), linear.predict_proba(x_val))},
    ])
    return dummy, linear, table.sort_values("f1_macro", ascending=False)


class MLP(nn.Module):
    def __init__(self, input_size, output_size=3, hidden_sizes=(64, 32), dropout=0.0):
        super().__init__()
        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(previous_size, hidden_size), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, output_size))
        self.network = nn.Sequential(*layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs):
        return self.network(inputs)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def train_model(config, x_train, y_train, x_val, y_val, class_weights, device, seed=42):
    max_epochs = config.get("max_epochs", 150)
    patience = config.get("patience", 15)
    fix_seed(seed)
    model = MLP(
        input_size=x_train.shape[1],
        hidden_sizes=config["hidden_sizes"],
        dropout=config["dropout"],
    ).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    train_x = torch.tensor(x_train, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_train.to_numpy(), dtype=torch.long, device=device)
    val_x = torch.tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val.to_numpy(), dtype=torch.long, device=device)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(train_x)
            val_logits = model(val_x)
            train_loss = loss_function(train_logits, train_y).item()
            val_loss = loss_function(val_logits, val_y).item()
            val_predictions = val_logits.argmax(1).cpu().numpy()
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1_macro": f1_score(y_val, val_predictions, average="macro"),
        })
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


def predict_model(model, x, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, device=device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return probabilities.argmax(1), probabilities
