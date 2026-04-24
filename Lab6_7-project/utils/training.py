import os
import time

import inspect
import functools

import pandas as pd
import torch
from typing import Type, TypeAlias, Any

from sklearn.model_selection import KFold
from torch.ao.pruning import scheduler
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler, LambdaLR

from utils.MLPClassifier import MLPClassifier





#Types
Criterion: TypeAlias = Type[CrossEntropyLoss | MSELoss]
Optimizer: TypeAlias = Type[SGD | Adam]
Scheduler: TypeAlias = Type[ReduceLROnPlateau | LRScheduler | LambdaLR]
Model: TypeAlias = Type[MLPClassifier]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get/Create Functions
# Prompt (Sonnet 4.6):
'''
Basing on following function structure, finish functions, do not change code of finished function. Refer to todos and structure.

```
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _parse_series(parameters:pd.Series) -> dict[str, Any]:
    best = dict(**parameters)
    all_params = {k.split('params_')[1]: v for k, v in best.items() if k.startswith('params_')}

    return all_params

def create_model(parameters:pd.Series, in_out_shape:tuple[int, int]) -> object:
    params_dict = _parse_series(parameters)

    model = MLPClassifier(
        input_dim=in_out_shape[0],
        output_dim=in_out_shape[1],
        n_hidden=params_dict['n_hidden'],
        hidden_dim=params_dict['hidden_dim']
    )

    return model



def get_optimzer(params:pd.Series, optimizer_class:Optimizer):

    pass


def get_scheduler(optimizer:torch.optim.Optimizer, params:pd.Series, scheduler_class:Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau) -> object:

    pass



def get_dataSet(X_trainval:pd.DataFrame, y_trainval:pd.Series | pd.DataFrame):
    X_trainval_tensor = torch.from_numpy(X_trainval.to_numpy()).float()
    y_trainval_tensor = torch.from_numpy(y_trainval.to_numpy()).float()

    dataset = TensorDataset(X_trainval_tensor, y_trainval_tensor)

    return dataset

def get_train_loaders(dataset:TensorDataset, train_idx, val_idx, batch_size:int=256, shuffle:bool=True):
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

    return train_loader, val_loader


def create_training_dict(parameters:pd.Series, in_out_shape:tuple[int, int], classes=None, n_folds:int = 5) -> dict[str, Any]:
    if classes is None:
        classes = {"here object name": "here class name form possible types to parse"} # as mutable taken from function above
    pass
    #TODO: Create a dict that will be easily pared (**) to train_one_fold function, n_folds should be list with fold id from 1-n_folds
    # Use above functions, remmber to parse each object (like schduler) params accordingly to possible schduler class an only with params avaviable in parameters

```

```
include edge case: betas for adam could be safed as beta1 and beta2
```

+ additional hand on writing
'''

def _parse_series(parameters:pd.Series) -> dict[str, Any]:
    best = dict(**parameters)
    all_params = {k.split('params_')[1]: v for k, v in best.items() if k.startswith('params_')}

    return all_params

def create_model(parameters:pd.Series, in_out_shape:tuple[int, int]) -> object:
    params_dict = _parse_series(parameters)

    model = MLPClassifier(
        input_dim=in_out_shape[0],
        output_dim=in_out_shape[1],
        n_hidden=params_dict['n_hidden'],
        hidden_dim=params_dict['hidden_dim']
    )

    return model


def _merge_betas(params_dict: dict) -> dict:
    """Merge beta1/beta2 keys into a single betas tuple if present."""
    beta1 = params_dict.pop('beta1', None)
    beta2 = params_dict.pop('beta2', None)

    if beta1 is not None and beta2 is not None:
        params_dict['betas'] = (beta1, beta2)
    elif beta1 is not None or beta2 is not None:
        raise ValueError(f"Only one of beta1/beta2 found — both are required to construct betas tuple.")

    return params_dict


def get_optimizer(params: pd.Series, optimizer_class: Optimizer) -> torch.optim.Optimizer:
    params_dict = _parse_series(params)
    params_dict = _merge_betas(params_dict)

    valid_keys = inspect.signature(optimizer_class.__init__).parameters.keys()
    optimizer_params = {k: v for k, v in params_dict.items() if k in valid_keys}

    return functools.partial(optimizer_class, **optimizer_params)


def get_scheduler(
        optimizer: torch.optim.Optimizer,
        params: pd.Series,
        scheduler_class: Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
) -> object:
    params_dict = _parse_series(params)

    valid_keys = inspect.signature(scheduler_class.__init__).parameters.keys()
    scheduler_params = {k: v for k, v in params_dict.items() if k in valid_keys}

    return scheduler_class(optimizer, **scheduler_params)

def get_dataSet(X_trainval:pd.DataFrame, y_trainval:pd.Series | pd.DataFrame):
    X_trainval_tensor = torch.from_numpy(X_trainval.to_numpy()).float()
    y_trainval_tensor = torch.from_numpy(y_trainval.to_numpy()).float()

    dataset = TensorDataset(X_trainval_tensor, y_trainval_tensor)

    return dataset

def get_train_loaders(dataset:TensorDataset, train_idx, val_idx, batch_size:int=256, shuffle:bool=True):
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

    return train_loader, val_loader


def create_training_dict(
    parameters: pd.Series,
    data: pd.DataFrame,
    target: pd.DataFrame,
    criterion: Criterion = CrossEntropyLoss,
    classes: dict = None,
    n_folds: int = 5,
    n_epochs:int = 100,
    max_norm: float = 1.0,
    write_model_dir: str = None,
    writer: SummaryWriter = None,
    random_state:int = 42
) -> dict[str, Any]:
    if classes is None:
        classes = {
            "optimizer_class": torch.optim.Adam,
            "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
        }

    params_dict = _parse_series(parameters)

    dataset = get_dataSet(X_trainval=data, y_trainval=target)

    return {
        "fold_id": list(range(1, n_folds + 1)),
        "n_epochs": n_epochs,
        "dataset": dataset,
        "batch_size": params_dict["batch_size"],
        "model_fn": lambda: create_model(parameters, in_out_shape=(data.shape[1], target.shape[1])),
        "optimizer_cls": get_optimizer(parameters, classes["optimizer_class"]),
        "scheduler_cls": lambda opt: get_scheduler(opt, parameters, classes["scheduler_class"]),
        "criterion": criterion(),
        "max_norm": max_norm,
        "write_model_dir": write_model_dir,
        "writer": writer,
        "random_state": random_state
    }


# Training Functions

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
            writer.add_scalar(f'train_loss/fold_{fold_id}', train_loss, epoch)
            writer.add_scalar(f'val_loss/fold_{fold_id}', val_loss, epoch)
            writer.add_scalar(f'learining_rate/fold{fold_id}', scheduler.get_last_lr()[0], epoch)
            writer.flush()

        if write_model_dir is not None:
            if best_val_loss > val_loss:
                best_val_loss = val_loss

                model_path = os.path.join(write_model_dir, f"fold_{fold_id}.pth")
                torch.save(model.state_dict(), model_path)

    avg_val_loss = running_val_loss / len(val_loader)


    return avg_val_loss

def train_from_dict(training_dict:dict[str, Any]):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(training_dict['dataset'])))):
        print(f'\n ===================== Training Fold {fold}  ===================== \n')
        start = time.time()

        # Loaders
        train_loader, val_loader = get_train_loaders(training_dict['dataset'], train_idx=train_idx, val_idx=val_idx,
                                                     batch_size=int(training_dict['batch_size']))

        # Model, Optimizer, Criterion
        criterion = training_dict['criterion']
        model = training_dict['model_fn']()
        optimizer = training_dict['optimizer_cls'](model.parameters())
        scheduler = training_dict['scheduler_cls'](optimizer)

        # Fold Training
        loss = train_one_fold(fold, model, train_loader, val_loader, optimizer, scheduler, criterion, n_epochs=training_dict['n_epochs'],
                              write_model_dir=training_dict['write_model_dir'], writer=training_dict['writer'])

        stop = time.time()
        print(f'validation loss = {loss:.4f}')
        print(f'Time: {stop-start:.2f} seconds')