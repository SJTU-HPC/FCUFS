import pandas as pd

def df_mean(df_data):
    mean = df_data
    return mean

def counters_to_features(df_data, time_slice):
    features = pd.DataFrame()

    # freq features
    if 'core_freq' in df_data.columns:
        features['core_freq'] = df_data['core_freq']
    if 'uncore_freq' in df_data.columns:
        features['uncore_freq'] = df_data['uncore_freq']

    # uop relevent features
    features['load_uop'] = df_mean(df_data['uop_load_num']) / df_mean(df_data['uop_num'])
    features['store_uop'] = df_mean(df_data['uop_store_num']) / df_mean(df_data['uop_num'])
    features['core_L2_miss'] = df_mean(df_data['core_L3_ref_num']) / df_mean(df_data['uop_num'])
    features['core_L3_miss'] = df_mean(df_data['core_L3_mis_num']) / df_mean(df_data['uop_num'])

    # output
    features['power'] = df_data['power'] / (time_slice * 1000000)
    features['uop/clock'] = df_data['uop_num'] / (time_slice * 2.49e+9)

    return features
