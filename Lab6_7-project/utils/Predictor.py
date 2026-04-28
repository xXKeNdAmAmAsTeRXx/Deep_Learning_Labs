from typing import Literal
import os

import numpy as np
import torch
from MLPClassifier import MLPClassifier

# TODO: Imports


# TODO: static Typing
class Predictor:
    def __init__(self, path:str, model_params:dict) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        folds_path = os.listdir(path)
        self.folds = []
        for f in folds_path:
            m = MLPClassifier(**model_params)
            m.load_state_dict(torch.load(os.path.join(path, f)))
            self.folds.append(m)

        self.in_out_dim = (model_params['input_dim'], model_params['output_dim'])

    def _mean_response_ensemble(self, data:np.ndarray) -> np.ndarray:
        data_t = torch.tensor(data)
        probs = np.zeros(self.in_out_dim[0])
        for idx, f in enumerate(self.folds):
            f.eval()
            with torch.no_grad():
                outputs = f(data_t)
                proba = torch.softmax(outputs, dim=0).numpy()

            mean_proba = np.mean(proba, axis=0)
            winner = mean_proba.argmax()

        return winner

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
