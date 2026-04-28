from typing import Literal
import json
import os

import numpy as np
import torch
from utils.MLPClassifier import MLPClassifier

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
        if ensemble == "mean_response":
            return self._mean_response_ensemble(data)
        else:
            return self._majority_voting_ensemble(data)

    def metric_report(self, data, target,ensemble:Literal['mean_response','majority_voting'] = "mean_response", plot_results:bool = True):
        # TODO: performe prediction and calculate metrics, show vizualizations
        pass

    def statistical_report(self, data, target, choosen_test:str,ensemble:Literal['mean_response','majority_voting'] = "mean_response", plot_results:bool = True):
        #TODO: function to wrap results of chosen statistical test
        pass
