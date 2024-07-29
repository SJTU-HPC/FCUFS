import numpy as np
import pandas as pd
import os
import glob
import sys
from tuning import tuning
from model import strategy
import json

moniter_path = "./sampler"
apps_dir = "../benchmarks"
cpu_info = {
    "core_num_per_socket": 20, 
    "socket_num": 2,
    "core_num": 40  # core_num_per_socket * socket_num
}

'''
strategy_id == -4: ondemand
strategy_id == -3: performance
strategy_id == -2: powersave
strategy_id == -1: maxmimum oncore and uncore frequencies
strategy_id == 0: minimize energy consumption with a performance loss constraint

frequency_scaling_strategy == 'UFS': only tuning uncore frequency
frequency_scaling_strategy == 'CFS': only tuning oncore frequency
frequency_scaling_strategy == 'FCUFS': tuning oncore and uncore frequencies at the core level
frequency_scaling_strategy == 'UCUFS': tuning oncore and uncore frequencies at the node level
'''
strategy_id = 0

core_freqs_scale = [2500, 2400, 2300, 2200, 2100, 2000, 1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000]  # core frequency [MHz]
uncore_freqs_scale = [2500, 2400, 2300, 2200, 2100, 2000, 1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000]     # uncore frequency [MHz]

def main():
    # parse configurations
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)
    frequency_scaling_strategy = config['stragety']
    perf_constraint = 1 - config['PL'] * 0.01
    cpu_info = {
        "core_num_per_socket": config['core_num_per_socket'], 
        "socket_num": config['socket_num'],
        "core_num": config['core_num_per_socket'] * config['socket_num']
    }
    sampling_interval = config['sampling_interval']
    oncore_freq_steps = config['oncore_frequencies_MHz']
    uncore_freq_steps = config['uncore_frequencies_MHz']

    # model path
    power_model_path = f"./trained_model/power_model_{frequency_scaling_strategy}.pth"
    perf_model_path = f"./trained_model/performance_model_{frequency_scaling_strategy}.pth"
    if frequency_scaling_strategy == "UCUFS":
        power_model_path = f"./trained_model/power_model_FCUFS.pth"
        perf_model_path = f"./trained_model/performance_model_FCUFS.pth"
    
    # launch tuning service
    mstrategy = strategy.strategy_setter(cpu_info, strategy_id, perf_constraint)
    tuning.launch_tuning(power_model_path, perf_model_path, cpu_info, mstrategy, oncore_freq_steps, uncore_freq_steps, moniter_path, sampling_interval, frequency_scaling_strategy)

if __name__ == "__main__":
    main()
