import torch
import time

class ModelLoader():
    def __init__(self, power_model_path, perf_model_path):
        self.power_model = torch.load(power_model_path)
        self.perf_model = torch.load(perf_model_path)
        self.power_model.eval()
        self.perf_model.eval()

    def predict(self, features):
        res = {}
        features = torch.tensor(features.values, dtype=torch.float)

        pred_power = self.power_model(features).reshape(features.size(0), -1)
        pred_power = pred_power / pred_power[0]
        pred_perf = self.perf_model(features).reshape(features.size(0), -1)
        pred_perf = pred_perf / pred_perf[0]
        pred_energy = pred_power / pred_perf

        res['pred_power'] = pred_power
        res['pred_perf'] = pred_perf
        res['pred_energy'] = pred_energy

        return res

