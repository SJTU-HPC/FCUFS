import numpy as np
import pandas as pd
import os
import time
import struct

def load_csv(file_path):
    return pd.read_csv(file_path).clip(lower=0)

def save_profile_data(data, save_dir, sub_dir, file_name, float_format=None):
    os.system("mkdir -p " + save_dir)
    os.system("mkdir -p " + save_dir + "/" + sub_dir)
    if float_format != None:
        data.to_csv(save_dir+"/"+sub_dir+"/"+file_name, index=False, float_format=float_format)
    else:
        data.to_csv(save_dir+"/"+sub_dir+"/"+file_name, index=False)

def np_save(file_name, arr, delimiter=','):
    np.savetxt(file_name, arr, delimiter=delimiter)

class strategy_setter:
    def __init__(self, cpu_info, strategy_id, arg):
        self.strategy_id = strategy_id
        self.arg = arg
        self.start_time = time.time()
        self.cpu_info = cpu_info

class hardware_setter:
    def __init__(self, cpu_info, core_freq_scale, uncore_freq_scale):
        self.core_num = cpu_info['core_num']
        self.socket_num = cpu_info['socket_num']
        self.core_num_per_socket = cpu_info['core_num_per_socket']
        self.max_core_freq = max(core_freq_scale)
        self.min_core_freq = min(core_freq_scale)
        self.max_uncore_freq = max(uncore_freq_scale)
        self.min_uncore_freq = min(uncore_freq_scale)
        self.power_f = []
        self.power_f.append(open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', 'r'))
        self.power_f.append(open('/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj', 'r'))
        self.power_f.append(open('/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj', 'r'))
        self.power_f.append(open('/sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj', 'r'))
        self.min_core_f = []
        self.max_core_f = []
        for core_id in range(self.core_num):
            self.min_core_f.append(open(f'/sys/devices/system/cpu/cpu{core_id}/cpufreq/scaling_min_freq', 'w'))
            self.max_core_f.append(open(f'/sys/devices/system/cpu/cpu{core_id}/cpufreq/scaling_max_freq', 'w'))
        self.uncore_f = []
        for socket_id in range(self.socket_num):
            self.uncore_f.append(open(f'/dev/cpu/{socket_id * self.core_num_per_socket}/msr', 'wb'))
            self.uncore_f[socket_id].seek(0x620)

    def __del__(self):
        for f in self.power_f:
            f.close()
        for f in self.min_core_f:
            f.close()
        for f in self.max_core_f:
            f.close()
        for f in self.uncore_f:
            f.close()

    def reset_core_frequency(self):
        for core_id in range(self.core_num):
            self.min_core_f[core_id].write(str(self.min_core_freq * 1000))
            self.max_core_f[core_id].write(str(self.max_core_freq * 1000))
            self.min_core_f[core_id].flush()
            self.max_core_f[core_id].flush()

    def reset_uncore_frequency(self):
        for socket_id in range(self.socket_num):
            uncore_freq_msr = (self.max_uncore_freq // 100) | ((self.min_uncore_freq // 100) << 8)
            self.uncore_f[socket_id].write(struct.pack('Q', uncore_freq_msr))
            self.uncore_f[socket_id].flush()

    def set_frequency(self, lower_core_freq, upper_core_freq, lower_uncore_freq, upper_uncore_freq):
        self.set_core_frequency(lower_core_freq, upper_core_freq)
        self.set_uncore_frequency(lower_uncore_freq, upper_uncore_freq)

    def set_max_frequency():
        lower_core_freq = [self.max_core_freq] * core_num
        upper_core_freq = [self.max_core_freq] * core_num
        lower_uncore_freq = [self.max_uncore_freq] * core_num
        upper_uncore_freq = [self.max_uncore_freq] * core_num
        self.set_core_frequency(lower_core_freq, upper_core_freq)
        self.set_uncore_frequency(lower_uncore_freq, upper_uncore_freq)

    def set_core_frequency(self, lower_core_freq, upper_core_freq):
        for core_id in range(self.core_num):
            self.min_core_f[core_id].write(str(lower_core_freq[core_id] * 1000))
            self.max_core_f[core_id].write(str(upper_core_freq[core_id] * 1000))
            self.min_core_f[core_id].flush()
            self.max_core_f[core_id].flush()

    def set_uncore_frequency(self, lower_uncore_freq, upper_uncore_freq):
        for socket_id in range(self.socket_num):
            uncore_freq_msr = (upper_uncore_freq[socket_id] // 100) | ((lower_uncore_freq[socket_id] // 100) << 8)
            self.uncore_f[socket_id].write(struct.pack('Q', uncore_freq_msr))
            self.uncore_f[socket_id].flush()

    def get_power(self):
        energy_uj = 0
        for f in self.power_f:
            energy_uj += int(f.readline().strip())
            f.seek(0)
        return energy_uj

