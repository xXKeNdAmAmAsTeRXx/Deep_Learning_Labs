from typing import Literal, Union
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils8.AudioCNN import AudioCNN
from sklearn.metrics import classification_report, RocCurveDisplay, confusion_matrix, mean_squared_error, \
    r2_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


class Predictor:
    def __init__(self, path: str, classification: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir_list = sorted(os.listdir(path))[::-1]

        dict_path = os.path.join(path, 'model_dict.json')
        with open(dict_path, 'r', encoding='utf-8-sig') as f:
            model_params = json.load(f)

        labels_path = os.path.join(path, 'data_labels.json')
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8-sig') as f:
                labels = json.load(f)
            self.labels_dict = {int(k): v for k, v in labels.items()}
        else:
            self.labels_dict = None

        folds_list = [s for s in model_dir_list if s.startswith('fold_')]

        self.folds = []
        for f in folds_list:
            fold_path = os.path.join(path, f)
            m = AudioCNN(**model_params)
            m.load_state_dict(torch.load(fold_path, map_location=self.device))
            m.to('cpu')
            self.folds.append(m)

        self.model_params = model_params
        self.classification = classification

    def _predict_in_chunks(self, model: torch.nn.Module, data: Union[np.ndarray, pd.DataFrame, Dataset],
                           chunk_size: int = 32) -> torch.Tensor:
        model.to(self.device)
        model.eval()

        if isinstance(data, Dataset):
            dataloader = DataLoader(data, batch_size=chunk_size, shuffle=False)
            outputs_list = []
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0].float().to(self.device)
                    else:
                        x = batch.float().to(self.device)
                    out = model(x)
                    outputs_list.append(out.cpu())
                    del x, out
                    torch.cuda.empty_cache()
            model.to('cpu')
            torch.cuda.empty_cache()
            return torch.cat(outputs_list, dim=0)

        if isinstance(data, pd.DataFrame):
            data = data.values

        outputs_list = []
        with torch.no_grad():
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                x = torch.from_numpy(chunk).float().to(self.device)
                out = model(x)
                outputs_list.append(out.cpu())
                del x, out
                torch.cuda.empty_cache()
        model.to('cpu')
        torch.cuda.empty_cache()
        return torch.cat(outputs_list, dim=0)

    def _mean_response_ensemble(self, data: Union[np.ndarray, pd.DataFrame, Dataset]) -> np.ndarray:
        all_probas = []
        for idx, f in enumerate(self.folds):
            outputs = self._predict_in_chunks(f, data)
            proba = torch.softmax(outputs, dim=1).numpy()
            all_probas.append(proba)

        mean_proba = np.mean(all_probas, axis=0)

        if self.classification:
            return mean_proba.argmax(axis=1)
        else:
            return mean_proba

    def _majority_voting_ensemble(self, data: Union[np.ndarray, pd.DataFrame, Dataset],
                                  return_votes: bool = False) -> np.ndarray:
        all_preds = []
        num_classes = None
        for f in self.folds:
            outputs = self._predict_in_chunks(f, data)
            if num_classes is None:
                num_classes = outputs.shape[1]
            fold_preds = torch.softmax(outputs, dim=1).numpy().argmax(axis=1)
            all_preds.append(fold_preds)
        all_preds = np.array(all_preds)

        majority_vote = np.array([np.bincount(col, minlength=num_classes).argmax() for col in all_preds.T])

        if return_votes:
            votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_classes), axis=0, arr=all_preds)
            return votes

        return majority_vote

    def predict_proba(self, data: Union[np.ndarray, pd.DataFrame, Dataset]):
        assert self.classification

        all_probas = []
        for idx, f in enumerate(self.folds):
            outputs = self._predict_in_chunks(f, data)
            proba = torch.softmax(outputs, dim=1).numpy()
            all_probas.append(proba)

        mean_proba = np.mean(all_probas, axis=0)

        return mean_proba

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dataset],
                ensemble: Literal['mean_response', 'majority_voting'] = "mean_response"):
        assert ensemble == 'mean_response' or self.classification

        if ensemble == "mean_response":
            return self._mean_response_ensemble(data)
        else:
            return self._majority_voting_ensemble(data)

    def metric_report(self, data: Union[np.ndarray, pd.DataFrame, Dataset],
                      target: Union[np.ndarray, pd.Series, None] = None,
                      ensemble: Literal['mean_response', 'majority_voting'] = "mean_response",
                      plot_results: bool = True) -> None | dict[str, float]:
        if target is None:
            assert isinstance(data, Dataset)
            dataloader = DataLoader(data, batch_size=len(data), shuffle=False)
            batch = next(iter(dataloader))
            assert isinstance(batch, (list, tuple)) and len(batch) > 1
            target = batch[1]
            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()

        if isinstance(target, pd.Series):
            target = target.values

        assert ensemble == 'mean_response' or self.classification

        if ensemble == 'mean_response':
            yhat = self._mean_response_ensemble(data)
        else:
            yhat = self._majority_voting_ensemble(data)

        if not self.classification:
            r2 = r2_score(target, yhat)
            mse = mean_squared_error(target, yhat)

            result = {'r2': r2, 'mse': mse}
            return result

        y_proba = self.predict_proba(data)
        report = classification_report(target, yhat, zero_division=0)

        if plot_results:
            fig, ax = plt.subplots(1, 2, figsize=(20, 6))
            ax = ax.flatten()

            cf_mtx = confusion_matrix(target, yhat)
            sns.heatmap(cf_mtx, annot=True, fmt="g", ax=ax[0])
            ax[0].set_title('Confusion Matrix')

            for idx in range(self.model_params['num_classes']):
                fpr, tpr, thresholds = roc_curve(target == idx, y_proba[:, idx])
                roc_auc = auc(fpr, tpr)
                display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
                display.plot(ax=ax[1])

            ax[1].plot([0, 1], [0, 1], color='white', linestyle='--')
            ax[1].set_title('Multiclass ROC Curve (One-vs-Rest)')

            if self.labels_dict is not None:
                labels = list(self.labels_dict.values())
                ax[0].set_xticklabels(labels)
                ax[0].set_yticklabels(labels)
                plt.legend(labels)
        plt.show()
        print(report)

        return None

    def human_pred(self, data: Union[np.ndarray, pd.DataFrame, Dataset],
                   ensemble: Literal['mean_response', 'majority_voting'] = "mean_response") -> np.ndarray:
        if ensemble == 'mean_response':
            yhat = self._mean_response_ensemble(data)
        else:
            yhat = self._majority_voting_ensemble(data)

        if not self.classification:
            return yhat

        if self.labels_dict is not None:
            map = self.labels_dict
            map_func = np.vectorize(map.get)
            pred = map_func(yhat)
        else:
            print("No labels provided")

        return pred