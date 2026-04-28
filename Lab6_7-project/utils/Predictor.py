from typing import Literal
import json
import os

import numpy as np
import pandas as pd
import torch
from utils.MLPClassifier import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, confusion_matrix, mean_squared_error, \
    r2_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Imports


# TODO: static Typing
class Predictor:
    def __init__(self, path:str, classification:bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir_list = sorted(os.listdir(path))[::-1]
        dict_path = os.path.join(path, model_dir_list.pop(0))

        with open(dict_path, 'r', encoding='utf-8-sig') as f:
            model_params = json.load(f)

        self.folds = []
        for f in model_dir_list:
            fold_path = os.path.join(path, f)
            m = MLPClassifier(**model_params)
            m.load_state_dict(torch.load(fold_path))
            self.folds.append(m)

        self.model_params = model_params
        self.classification = classification

    def _mean_response_ensemble(self, data:np.ndarray) -> np.ndarray:
        data_t = torch.from_numpy(data).float().to(self.device)

        all_probas = []
        for idx, f in enumerate(self.folds):
            f.to(self.device)
            f.eval()
            with torch.no_grad():
                outputs = f(data_t)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probas.append(proba)

        mean_proba = np.mean(all_probas, axis=0)

        if self.classification:
            return mean_proba.argmax(axis=1)
        else:
            return mean_proba

    def _majority_voting_ensemble(self, data: np.ndarray, return_votes:bool=False) -> np.ndarray:
        data_t = torch.from_numpy(data).float().to(self.device)

        all_preds = []
        for f in self.folds:
            f.to(self.device)
            f.eval()
            with torch.no_grad():
                outputs = f(data_t)
                fold_preds = torch.softmax(outputs, dim=1).cpu().numpy().argmax(axis=1)
                all_preds.append(fold_preds)
        all_preds = np.array(all_preds)

        majority_vote = np.array([np.bincount(col, minlength=outputs.shape[1]).argmax() for col in all_preds.T])

        if return_votes:
            votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=outputs.shape[1]), axis=0, arr=all_preds)
            return votes

        return majority_vote

    def predict_proba(self, data):
        assert self.classification
        data_t = torch.from_numpy(data).float().to(self.device)

        all_probas = []
        for idx, f in enumerate(self.folds):
            f.to(self.device)
            f.eval()
            with torch.no_grad():
                outputs = f(data_t)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probas.append(proba)

        mean_proba = np.mean(all_probas, axis=0)

        return mean_proba



    def predict(self, data, ensemble:Literal['mean_response','majority_voting'] = "mean_response"):
        assert ensemble == 'mean_response' or self.classification

        if ensemble == "mean_response":
            return self._mean_response_ensemble(data)
        else:
            return self._majority_voting_ensemble(data)

    def metric_report(self, data, target,ensemble:Literal['mean_response','majority_voting'] = "mean_response", plot_results:bool = True) -> None:

        assert  ensemble == 'mean_response' or self.classification

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
        report = classification_report(target, yhat)


        if plot_results:

            fig, ax = plt.subplots(1,2, figsize=(20, 6))
            ax = ax.flatten()

            cf_mtx = confusion_matrix(target, yhat)
            sns.heatmap(cf_mtx, annot=True, fmt="g", ax=ax[0])

            for idx in range(self.model_params['output_dim']):
                fpr, tpr, thresholds = roc_curve(target == idx, y_proba[:, idx])
                roc_auc = auc(fpr, tpr)
                display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
                display.plot(ax=ax[1])

            ax[1].plot([0, 1], [0, 1], color='white', linestyle='--')
            ax[1].set_title('Multiclass ROC Curve (One-vs-Rest)')
            plt.show()

        print(report)

        return None

    def statistical_report(self, data, target, choosen_test:str,ensemble:Literal['mean_response','majority_voting'] = "mean_response", plot_results:bool = True):
        #TODO: function to wrap results of chosen statistical test
        pass
