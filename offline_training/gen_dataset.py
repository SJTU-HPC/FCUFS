import numpy as np
import pandas as pd
from utils import utils
import glob
import os

def get_benchmark_bins(benchmarks_dir, benchmarks_name):
    bins = list()
    for benchmark_name in benchmarks_name:
        bins.extend(glob.glob(os.path.join(benchmarks_dir, benchmark_name)))
    return bins

def get_data(data_dir, data_name_list):
    data = pd.DataFrame()
    path_list = get_benchmark_bins(data_dir, data_name_list)
    for path in path_list:
        print(path)
        data = pd.concat([data, utils.load_csv(path + "/features.csv")])
    return data

strategies = ["FCUFS"]
train_set = ["bt*", "cg*", "ep*", "ft*", "lu*", "mg*", "sp*", "ua*"]

for strategy in strategies:
    data_dir = f"./trainset_0415_{strategy}"
    save_dir = data_dir
    train_data = get_data(data_dir, train_set)
    if 'label' in train_data.columns:
        train_data = train_data.drop('label', axis=1)
    utils.save_profile_data(train_data, save_dir, "dataset", "train.csv", '%.6f')
