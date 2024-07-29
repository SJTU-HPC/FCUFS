import pandas as pd

def df_mean(df_data):
    mean = df_data.mean()
    return mean

def counters_to_features_infer(df_data, time_slice):
    features = pd.DataFrame()
    # uop_num,br_num,br_miss_num,uop_load_num,uop_store_num,core_L3_ref_num,core_L3_mis_num,L3_mis_num,power,core_freq,uncore_freq
    if 'core_freq' in df_data:
        features['core_freq'] = df_data['core_freq']
    if 'uncore_freq' in df_data:
        features['uncore_freq'] = df_data['uncore_freq']

    # uop relevent features
    features['load_uop'] = df_data['uop_load_num'] / df_data['uop_num']
    features['store_uop'] = df_data['uop_store_num'] / df_data['uop_num']
    features['core_L2_miss'] = df_data['core_L3_ref_num'] / df_data['uop_num']
    features['core_L3_miss'] = df_data['core_L3_mis_num'] / df_data['uop_num']

    return features
