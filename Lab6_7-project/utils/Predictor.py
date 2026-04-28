from typing import Literal
import json
import os

import numpy as np
import torch
from MLPClassifier import MLPClassifier

# TODO: Imports


# TODO: static Typing
class Predictor:
    def __init__(self, path:str, classification:bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir_list = sorted(os.listdir(path))
        dict_path = os.path.join(path, model_dir_list.pop(0))
        with open(dict_path, 'r') as f:
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
        data_t = torch.tensor(data)
        probs = np.zeros(data.shape[0], self.model_params['output_dim'])
        for idx, f in enumerate(self.folds):
            f.eval()
            with torch.no_grad():
                outputs = f(data_t)
                proba = torch.softmax(outputs, dim=0).numpy()

            probs[idx] = proba

        mean_proba = np.mean(proba, axis=0)

        if self.classification:
            return mean_proba.argmax(axis=0)
        else:
            return mean_proba

    def _majority_voting_ensemble(self):
        pass

    def predict_proba(self,data, ensemble:Literal['mean_response','majority_voting'] = "mean_response"):
        # TODO: Predction on given data returning probability
        pass

    def predict(self, data, ensemble:Literal['mean_response','majority_voting'] = "mean_response"):
        # TODO: Prediction on Given data with chosen ensemble method
        pass

    def metric_report(self, data, target,ensemble:Literal['mean_response','majority_voting'] = "mean_response", plot_results:bool = True):
        # TODO: performe prediction and calculate metrics, show vizualizations
        pass

    def statistical_report(self, data, target, choosen_test:str,ensemble:Literal['mean_response','majority_voting'] = "mean_response", plot_results:bool = True):
        #TODO: function to wrap results of chosen statistical test
        pass
