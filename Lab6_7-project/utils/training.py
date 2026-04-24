import os

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Subset

from utils.MLPClassifier import MLPClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(parameters:pd.Series, in_out_shape:tuple[int, int]) -> object:
    model = MLPClassifier(
        input_dim=in_out_shape[0],
        output_dim=in_out_shape[1],
        n_hidden=parameters['params_n_hidden'],
        hidden_dim=parameters['params_hidden_dim']
    )

    return model

def get_dataSet(X_trainval:pd.DataFrame, y_trainval:pd.Series | pd.DataFrame):
    X_trainval_tensor = torch.from_numpy(X_trainval.to_numpy()).float()
    y_trainval_tensor = torch.from_numpy(y_trainval.to_numpy()).float()

    dataset = TensorDataset(X_trainval_tensor, y_trainval_tensor)

    return dataset

def get_train_loaders(dataset:TensorDataset, train_idx, val_idx, batch_size:int=256, shuffle:bool=True):
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

    return train_loader, val_loader



def _train_one_epoch(model, loader, optimizer, criterion, max_norm:float=1.0):
    model.train()
    running_loss = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        running_loss += loss.item()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
    avg_loss = running_loss / len(loader)

    return avg_loss

def _calculate_val_loss(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X,y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = model(X)
            loss = criterion(output, y)
            running_loss += loss.item()
    avg_loss = running_loss / len(loader)

    return avg_loss


def train_one_fold(fold_id:int, model, training_loader:DataLoader, val_loader:DataLoader, optimizer, scheduler, criterion, n_epochs:int, max_norm:float=1.0, write_model_dir:str=None, writer:SummaryWriter=None):
    model.to(DEVICE)

    best_val_loss = float("inf")
    running_val_loss = 0.0
    for epoch in range(n_epochs):
        train_loss = _train_one_epoch(model, training_loader, optimizer, criterion, max_norm)
        val_loss = _calculate_val_loss(model, val_loader, criterion)
        running_val_loss += val_loss

        scheduler.step(val_loss)

        if writer is not None:
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.flush()

        if write_model_dir is not None:
            if best_val_loss > val_loss:
                best_val_loss = val_loss

                model_path = os.path.join(write_model_dir, f"fold_{fold_id}.pth")
                torch.save(model.state_dict(), model_path)

    avg_val_loss = running_val_loss / len(val_loader)


    return avg_val_loss

