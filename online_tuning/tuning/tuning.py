import os
import pandas as pd
import threading
import time
from utils import utils
from model import inference
from tuning import gen_features
import subprocess
import multiprocessing

def obtain_benchmarks(directory):
    return [os.path.join(directory, entry) for entry in os.listdir(directory)]

def remove_prefix(path):
    filename_with_extension = os.path.basename(path)
    return os.path.splitext(filename_with_extension)[0]

class freq_tuner():
    def __init__(self, strategy, cpu_info, core_freqs_scale, uncore_freqs_scale, power_model_path, perf_model_path, moniter_path, time_slice, frequency_scaling_strategy):
        super().__init__()
        self.core_freqs_scale = core_freqs_scale
        self.uncore_freqs_scale = uncore_freqs_scale
        self.model = inference.ModelLoader(power_model_path, perf_model_path)
        self.strategy = strategy
        self.result = None
        self.energy = 0
        self.time_slice = time_slice
        self.moniter_path = moniter_path
        self.cpu_info = cpu_info
        self.fc_strategy = frequency_scaling_strategy
        self.hardware_setter = utils.hardware_setter(cpu_info, core_freqs_scale, uncore_freqs_scale)

    def run(self):
        run_log = []
        collected_data = list()
        process = subprocess.Popen([self.moniter_path, str(self.time_slice)], stdout=subprocess.PIPE)

        if self.fc_strategy == "CFS": # CFS
            freq_comb = pd.DataFrame(self.core_freqs_scale, columns=['core_freq'])
            freq_comb_norm = freq_comb.copy()
            freq_comb_norm['core_freq'] = (freq_comb_norm['core_freq'] - freq_comb_norm['core_freq'].min()) / (freq_comb_norm['core_freq'].max() - freq_comb_norm['core_freq'].min())
        elif self.fc_strategy == "UFS": # UFS
            freq_comb = pd.DataFrame(self.uncore_freqs_scale, columns=['uncore_freq'])
            freq_comb_norm = freq_comb.copy()
            freq_comb_norm['uncore_freq'] = (freq_comb_norm['uncore_freq'] - freq_comb_norm['uncore_freq'].min()) / (freq_comb_norm['uncore_freq'].max() - freq_comb_norm['uncore_freq'].min())
        else:   # CUFS
            freq_comb = pd.DataFrame([(d, e) for d in self.core_freqs_scale for e in self.uncore_freqs_scale], columns=['core_freq', 'uncore_freq'])
            freq_comb_norm = freq_comb.copy()
            freq_comb_norm['core_freq'] = (freq_comb_norm['core_freq'] - freq_comb_norm['core_freq'].min()) / (freq_comb_norm['core_freq'].max() - freq_comb_norm['core_freq'].min())
            freq_comb_norm['uncore_freq'] = (freq_comb_norm['uncore_freq'] - freq_comb_norm['uncore_freq'].min()) / (freq_comb_norm['uncore_freq'].max() - freq_comb_norm['uncore_freq'].min())

        max_energy = get_max_energy()
        last_energy = get_cur_energy()

        # unset frequencies
        self.hardware_setter.reset_core_frequency()
        self.hardware_setter.reset_uncore_frequency()

        if self.strategy.strategy_id == -4: # ondemand
            os.system("cpupower frequency-set -g ondemand")

        if self.strategy.strategy_id == -3: # performance
            os.system("cpupower frequency-set -g performance")

        if self.strategy.strategy_id == -2: # powersave
            os.system("cpupower frequency-set -g powersave")

        core_num = self.cpu_info["core_num"]
        cur_core_freq = pd.DataFrame({'core_freq': [max(self.core_freqs_scale)] * core_num})

        while True:
            line = process.stdout.readline()
            if not line:
                break

            cur_energy = get_cur_energy()
            self.energy += calc_total_energy(last_energy, cur_energy, max_energy)
            last_energy = cur_energy

            # parse data
            collected_data = line.decode()
            dict_item = {}
            for pair in collected_data.split(","):
                key, value = pair.split(":")
                if key == 'power':
                    cur_power = int(value) / self.time_slice
                    continue
                keys = dict_item.keys()
                if key not in keys:
                    dict_item[key] = []
                dict_item[key].append(int(value))

            collected_data = pd.DataFrame(dict_item)

            # get cpu util
            cpu_util = collected_data['cpu_cycle'] / (cur_core_freq['core_freq'] * 1000000) / (self.time_slice / 1000000)
            collected_data = collected_data.drop('cpu_cycle', axis=1)

            if self.fc_strategy == "UCUFS":
                collected_data = collected_data.mean(axis = 0).to_frame().transpose()

            # gen features
            collected_data = collected_data.merge(freq_comb_norm, how='cross')                
            features = gen_features.counters_to_features_infer(collected_data, self.time_slice / 1000000)

            # predict perf and power under different core and uncore frequency
            predict_res = self.model.predict(features)

            # choose core and uncore frequency
            cur_log, cur_core_freq = self.strategy.choose_freq(predict_res, cpu_util, freq_comb.to_numpy(), cur_power, cur_core_freq, self.fc_strategy, self.hardware_setter)

def get_max_energy():
    files = ['/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj',
            '/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/max_energy_range_uj',
            '/sys/class/powercap/intel-rapl/intel-rapl:1/max_energy_range_uj',
            '/sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/max_energy_range_uj']
    max_energy = [0, 0, 0, 0]
    i = 0
    for file in files:
        with open(file, 'r') as file:
            max_energy[i] = int(file.readline())
        i = i + 1

    return max_energy

def calc_total_energy(last_energy, cur_energy, max_energy):
    total_energy = 0
    for i in range(4):
        if cur_energy[i] >= last_energy[i]:
            total_energy += cur_energy[i] - last_energy[i]
        else:
            total_energy += max_energy[i] - last_energy[i] + cur_energy[i]

    return total_energy

def get_cur_energy():
    files = ['/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj',
            '/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj',
            '/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj',
            '/sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj']
    energy = [0, 0, 0, 0]
    i = 0
    for file in files:
        with open(file, 'r') as file:
            energy[i] = int(file.readline())
        i = i + 1

    return energy

def launch_tuning(power_model_path, perf_model_path, cpu_info, strategy, core_freqs_scale, uncore_freqs_scale, moniter_path, time_slice, frequency_scaling_strategy):
    _freq_tuner = freq_tuner(strategy, cpu_info, core_freqs_scale, uncore_freqs_scale, power_model_path, perf_model_path, moniter_path, time_slice * 1000000, frequency_scaling_strategy)

    _freq_tuner.run()
