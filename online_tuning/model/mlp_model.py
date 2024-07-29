import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from utils import utils

# Define the MLP model
class MLP(nn.Module):
    non_nan_columns = None
    max_vals  = None
    min_vals = None

    def __init__(self, input_size, hidden_size, output_size, hidden_layer_num):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.act_f = nn.ReLU()
        for _ in range(hidden_layer_num - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            if i == 0:
                x = self.act_f(self.layers[i](x) + x)
            else:
                x = self.act_f(self.layers[i](x) + x)
        x = self.layers[len(self.layers) - 1](x)

        return x

    def filter_nan_inf(self, features):
        mask = torch.isnan(features).any(dim=1)
        self.non_nan_columns = ~mask
        features = features[self.non_nan_columns, :]

        mask = torch.isinf(features).any(dim=1)
        self.non_inf_columns = ~mask
        features = features[self.non_inf_columns, :]
        return features

    def pre_process(self, features):
        return features

    def de_process(self, features):
        return features

    def load_features(self, path_to_features, target):
        features = utils.load_csv(path_to_features)

        if target == "power":
            features.drop('uop/clock', axis=1, inplace=True)
        elif target == "performance":
            features.drop('power', axis=1, inplace=True)
        features = torch.tensor(features.values, dtype=torch.float32)
        features = self.filter_nan_inf(features)

        x = features[:, 0:-1]
        y = features[:, -1:]

        return self.pre_process(x), y

def get_features_num(path_to_features, target):
    features = utils.load_csv(path_to_features)

    if target == "power":
        features.drop('uop/clock', axis=1, inplace=True)
    elif target == "performance":
        features.drop('power', axis=1, inplace=True)
    features = torch.tensor(features.values, dtype=torch.float32)
    x = features[:, 0:-1]
    return x.size(1)
