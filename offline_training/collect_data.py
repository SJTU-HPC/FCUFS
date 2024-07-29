import numpy as np
import pandas as pd
import os
import glob
import itertools
import sys
from utils import utils
from benchmark import benchmark
from benchmark import gen_features

moniter_path = "./sampler"
benchmarks_dir = "../benchmarks"

benchmarks_used = [
    "../benchmarks/NPB_OMP_D/ua.sh",
    "../benchmarks/NPB_OMP_D/lu.sh",
    "../benchmarks/NPB_OMP_D/ep.sh",
    "../benchmarks/NPB_OMP_D/ft.sh",
    "../benchmarks/NPB_OMP_D/mg.sh",
    "../benchmarks/NPB_OMP_D/cg.sh",
    "../benchmarks/NPB_OMP_D/bt.sh",
    "../benchmarks/NPB_OMP_D/sp.sh",
]

cpu_info = {
    "core_num_per_socket": 20, 
    "socket_num": 2,
    "core_num": 40
}

strategies = ["FCUFS"]
core_freqs = [2500, 2400, 2300, 2200, 2100, 2000, 1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000]  # core frequency [MHz]
uncore_freqs = [2500, 2400, 2300, 2200, 2100, 2000, 1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000]     # uncore frequency [MHz]
time_slice = 1 #s

def process_data(data):
    parsed_data = []
    for item in data:
        dict_item = {}
        for pair in item.split(","):
            key, value = pair.split(":")
            dict_item[key] = int(value)
        parsed_data.append(dict_item)
    data = pd.DataFrame(parsed_data)

    return data

def main():
    hardware_setter = utils.hardware_setter(cpu_info, core_freqs, uncore_freqs)
    hardware_setter.reset_core_frequency()
    hardware_setter.reset_uncore_frequency()

    for strategy in strategies:
        save_dir = f"./trainset_0415_{strategy}"
        for benchmark_path in benchmarks_used:
            features = pd.DataFrame()
            raw_data = pd.DataFrame()
            benchmark_name = benchmark.remove_prefix(benchmark_path)
            print("[INFO] ************** collecting", benchmark_name, "data **************")

            if strategy == "CFS":
                hardware_setter.reset_uncore_frequency()
                for core_freq in core_freqs:
                    print("[INFO] collecting core frequency:", core_freq, "MHz")
                    hardware_setter.set_core_frequency([core_freq] * cpu_info["core_num"], [core_freq] * cpu_info["core_num"])
                    cur_data = process_data(benchmark.run_benchmark(benchmark_path, moniter_path, time_slice))

                    # record data
                    cur_data['core_freq'] = core_freq
                    raw_data = pd.concat([raw_data, cur_data])

            if strategy == "UFS":
                hardware_setter.reset_core_frequency()
                for uncore_freq in uncore_freqs:
                    print("[INFO] collecting uncore frequency:", uncore_freq, "MHz")
                    hardware_setter.set_uncore_frequency([uncore_freq] * cpu_info["core_num"], [uncore_freq] * cpu_info["core_num"])
                    cur_data = process_data(benchmark.run_benchmark(benchmark_path, moniter_path, time_slice))

                    # record data
                    cur_data['uncore_freq'] = uncore_freq
                    raw_data = pd.concat([raw_data, cur_data])

            if strategy == "FCUFS": 
                for core_freq, uncore_freq in itertools.product(core_freqs, uncore_freqs):
                    print("[INFO] collecting core frequency:", core_freq, "MHz", "uncore frequency:", uncore_freq, "MHz")
                    hardware_setter.set_frequency([core_freq] * cpu_info["core_num"], [core_freq] * cpu_info["core_num"], [uncore_freq] * cpu_info["core_num"], [uncore_freq] * cpu_info["core_num"])
                    cur_data = process_data(benchmark.run_benchmark(benchmark_path, moniter_path, time_slice))

                    # record data
                    cur_data['core_freq'] = core_freq
                    cur_data['uncore_freq'] = uncore_freq
                    raw_data = pd.concat([raw_data, cur_data])

            utils.save_profile_data(raw_data, save_dir, benchmark_name, 'raw_data.csv')

if __name__ == "__main__":
    main()
