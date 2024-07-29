import numpy as np
import pandas as pd
from utils import utils
from benchmark import gen_features
from sklearn.cluster import DBSCAN
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch

feature_num = 4
strategies = ["FCUFS"]
dataset = ["bt", "cg", "ep", "ft", "lu", "mg", "sp", "ua"]

def get_benchmark_bins(benchmarks_dir, benchmarks_name):
    bins = list()
    for benchmark_name in benchmarks_name:
        bins.extend(glob.glob(os.path.join(benchmarks_dir, benchmark_name)))
    return bins

def get_data(data_dir, data_name_list):
    data = dict()
    path_list = get_benchmark_bins(data_dir, data_name_list)
    for path in path_list:
        data[path] = utils.load_csv(path + "/raw_data.csv")
    return data

def filter(df, core_freq, uncore_freq):
    if core_freq != None:
        df = df[df['core_freq'] == core_freq]
    if uncore_freq != None:
        df = df[df['uncore_freq'] == uncore_freq]
    quantile_value = df['uop_num'].mean()
    df = df[df['uop_num'] >= quantile_value * 0.2]
    return df

for strategy in strategies:
    data_dir = f"./trainset_0415_{strategy}"
    save_dir = data_dir
    counter_dataset = get_data(data_dir, dataset)
    print(strategy)
    for path, raw_data in counter_dataset.items():
        print(path)
        filter_raw_data = pd.DataFrame()
        if strategy == "CFS":
            feature_start = 1
            core_freqs = raw_data['core_freq'].unique()
            for core_freq in core_freqs:
                res = filter(raw_data, core_freq, None)
                filter_raw_data = pd.concat([res, filter_raw_data])
        elif strategy == "UFS":
            feature_start = 1
            uncore_freqs = raw_data['uncore_freq'].unique()
            for uncore_freq in uncore_freqs:
                res = filter(raw_data, None, uncore_freq)
                filter_raw_data = pd.concat([res, filter_raw_data])
        elif strategy == "CUFS":
            feature_start = 2
            core_freqs = raw_data['core_freq'].unique()
            uncore_freqs = raw_data['uncore_freq'].unique()
            for core_freq in core_freqs:
                for uncore_freq in uncore_freqs:
                    res = filter(raw_data, core_freq, uncore_freq)
                    filter_raw_data = pd.concat([res, filter_raw_data])

        data = gen_features.counters_to_features(filter_raw_data, 1)

        features = data.iloc[:, feature_start : feature_start + feature_num]
        features = features.round(5)
        features = features.apply(lambda x: x / x.mean())

        # classification
        class_model = Birch(n_clusters=None, threshold=0.2, branching_factor=20)

        dbscan_labels = class_model.fit_predict(features)
        data["label"] = dbscan_labels

        # filter class
        data = data[data['label'] != -1]

        # normalize performance and power
        if strategy == "CFS":
            filt_data = data[data['core_freq'] == data['core_freq'].max()]
            labels_to_filt = filt_data['label'].unique()
        elif strategy == "UFS":
            filt_data = data[data['uncore_freq'] == data['uncore_freq'].max()]
            labels_to_filt = filt_data['label'].unique()
        else:
            filt_data = data[data['core_freq'] == data['core_freq'].max()]
            filt_data = filt_data[filt_data['uncore_freq'] == filt_data['uncore_freq'].max()]
            labels_to_filt = filt_data['label'].unique()

        normalized_data = pd.DataFrame()

        for label_filt in labels_to_filt:
            filt_data = data[data['label'] == label_filt]
            if strategy == "CFS":
                if len(filt_data) < len(core_freqs) * 4:
                    continue
                mean_data = filt_data.groupby('core_freq').mean().reset_index()
                if len(mean_data) < len(core_freqs):
                    continue
                mean_data['power'] = mean_data['power'] / mean_data['power'].iloc[-1]
                mean_data['uop/clock'] = mean_data['uop/clock'] / mean_data['uop/clock'].iloc[-1]
            if strategy == "UFS":
                mean_data = filt_data.groupby('uncore_freq').mean().reset_index()
                if len(mean_data) < len(uncore_freqs):
                    continue
                mean_data['power'] = mean_data['power'] / mean_data['power'].iloc[-1]
                mean_data['uop/clock'] = mean_data['uop/clock'] / mean_data['uop/clock'].iloc[-1]
            if strategy == "CUFS":
                mean_data = filt_data.groupby(['core_freq', 'uncore_freq']).mean().reset_index()
                if len(mean_data) < len(uncore_freqs) * len(core_freqs):
                    continue
                mean_data['power'] = mean_data['power'] / mean_data['power'].iloc[-1]
                mean_data['uop/clock'] = mean_data['uop/clock'] / mean_data['uop/clock'].iloc[-1]

            normalized_data = pd.concat([mean_data, normalized_data])

        if "core_freq" in normalized_data.columns:
            normalized_data['core_freq'] = (normalized_data['core_freq'] - normalized_data['core_freq'].min()) / (normalized_data['core_freq'].max() - normalized_data['core_freq'].min())
        if "uncore_freq" in normalized_data.columns:
            normalized_data['uncore_freq'] = (normalized_data['uncore_freq'] - normalized_data['uncore_freq'].min()) / (normalized_data['uncore_freq'].max() - normalized_data['uncore_freq'].min())

        utils.save_profile_data(normalized_data, save_dir, path.split('/')[-1], 'features.csv', '%.6f')
