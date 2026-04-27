from typing import Literal

import torch
# TODO: Imports


# TODO: static Typing
class Predictor:
    def __init__(self):
        # TODO: Write init function - keep models or model paths
        pass


    def _mean_response_ensemble(self):
        pass

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
