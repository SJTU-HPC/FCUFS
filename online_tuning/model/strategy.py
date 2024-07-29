from datetime import datetime
import torch
from utils import utils
import time
import numpy as np
import pandas as pd

'''
strategy_id == -4: ondemand
strategy_id == -3: performance
strategy_id == -2: powersave
strategy_id == -1: maxmimum oncore and uncore frequencies
strategy_id == 0: minimize energy consumption with a performance loss constraint
'''

cpu_util_s = 0.2

class strategy_setter:
    def __init__(self, cpu_info, strategy_id, arg):
        self.strategy_id = strategy_id
        self.arg = arg
        self.start_time = time.time()
        self.cpu_info = cpu_info

    # pred_data['pred_power'](%)
    # pred_data['pred_perf']
    # pred_data['pred_energy']
    def choose_freq(self, pred_data, cpu_util, freqs, cur_power, cur_core_freq, fc_strategy, hardware_setter):
        core_num = self.cpu_info["core_num"]
        socket_num = self.cpu_info["socket_num"]
        core_num_per_socket = self.cpu_info["core_num_per_socket"]
        now = round((time.time() - self.start_time) * 1000)

        if self.strategy_id < 0:
            log = {}
            print(f"[INFO] {now}ms,", f"current power {round(cur_power, 2)}w")
            log['time'] = now
            return log, cur_core_freq

################################################## CFS ##################################################
        if fc_strategy == "CFS" and self.strategy_id == 0:
            pred_data['pred_perf'] = pred_data['pred_perf'].reshape(-1, len(freqs)) # core_num x freq_num
            pred_data['pred_power'] = pred_data['pred_power'].reshape(-1, len(freqs)) # core_num x freq_num
            max_core_freq = max(freqs[:, 0])
            min_core_freq = min(freqs[:, 0])
            lower_core_freq = [min_core_freq] * core_num
            upper_core_freq = [max_core_freq] * core_num
            # obtain buzy cores
            buzy_core = (cpu_util[cpu_util > cpu_util_s].index).tolist()

            mask = pred_data['pred_perf'] >= self.arg
            if torch.nonzero(mask).size(0) == 0:
                for core_id in buzy_core:
                    lower_core_freq[core_id] = max_core_freq
            else:
                pred_energy_tmp = torch.where(mask, pred_data['pred_power'], torch.full_like(pred_data['pred_power'], 10000))
                min_values, min_indices = torch.min(pred_energy_tmp, dim=1)
                min_indices = min_indices.tolist()
                for core_id in buzy_core:
                    lower_core_freq[core_id] = freqs[min_indices[core_id], 0]
                    upper_core_freq[core_id] = freqs[min_indices[core_id], 0]

            print(f"[INFO] {now}ms, Set core0 to {lower_core_freq[0]} MHz")
            hardware_setter.set_core_frequency(lower_core_freq, upper_core_freq)
            log = {}
            log['time'] = now
            log['core'] = lower_core_freq

            return log, pd.DataFrame({'core_freq': upper_core_freq})


################################################## UFS ##################################################
        if fc_strategy == "UFS" and self.strategy_id == 0:
            pred_data['pred_perf'] = pred_data['pred_perf'].reshape(-1, len(freqs)) # core_num x freq_num
            pred_data['pred_power'] = pred_data['pred_power'].reshape(-1, len(freqs)) # core_num x freq_num

            max_uncore_freq = max(freqs[:, 0])
            min_uncore_freq = min(freqs[:, 0])
            lower_uncore_freq = [min_uncore_freq] * socket_num
            upper_uncore_freq = [max_uncore_freq] * socket_num
            uncore_freq_num = np.shape(freqs)[0]

            # obtain buzy cores
            buzy_core = (cpu_util[cpu_util > cpu_util_s].index).tolist()
            buzy_core_num_per_socket = [0] * socket_num
            for core_id in buzy_core:
                buzy_core_num_per_socket[core_id // core_num_per_socket] += 1

            # search optimal uncore frequencys
            for socket_id in range(socket_num):
                if buzy_core_num_per_socket[socket_id] == 0:
                    continue
                opt_energy = 1000
                opt_perf = 0
                for uncore_freq_id in range(uncore_freq_num):
                    cur_uncore_freq = freqs[uncore_freq_id, 0]
                    perf = 0
                    energy = 0
                    for core_id in buzy_core:
                        if core_id // core_num_per_socket == socket_id:
                            perf += pred_data['pred_perf'][core_id, uncore_freq_id].item()
                            energy += pred_data['pred_power'][core_id, uncore_freq_id].item()
                    perf /= buzy_core_num_per_socket[socket_id]
                    energy /= buzy_core_num_per_socket[socket_id]
                    energy = round(energy * 100)

                    if perf >= self.arg:
                        if energy <= opt_energy:
                            opt_energy = energy
                            lower_uncore_freq[socket_id] = cur_uncore_freq
                            upper_uncore_freq[socket_id] = cur_uncore_freq

            print(f"[INFO] {now}ms, Set uncore0 to {lower_uncore_freq[0]}-{upper_uncore_freq[0]} MHz, uncore1 to {lower_uncore_freq[1]}-{upper_uncore_freq[1]} MHz")
            hardware_setter.set_uncore_frequency(lower_uncore_freq, upper_uncore_freq)
            log = {}
            log['time'] = now
            log['uncore'] = lower_uncore_freq

            return log, cur_core_freq

################################################## CUFS ##################################################

        min_core_freq = min(freqs[:, 0])
        max_core_freq = max(freqs[:, 0])
        min_uncore_freq = min(freqs[:, 1])
        max_uncore_freq = max(freqs[:, 1])
        core_freq_num = len(np.unique(freqs[:, 0]).tolist())
        uncore_freq_num = len(np.unique(freqs[:, 1]).tolist())

        log = {}
        pred_data['pred_perf'] = pred_data['pred_perf'].reshape(-1, len(freqs)) # core_num x freq_num
        pred_data['pred_power'] = pred_data['pred_power'].reshape(-1, len(freqs)) # core_num x freq_num
        pred_data['pred_energy'] = pred_data['pred_energy'].reshape(-1, len(freqs)) # core_num x freq_num

        lower_core_freq = [min_core_freq] * core_num
        upper_core_freq = [max_core_freq] * core_num
        lower_uncore_freq = [min_uncore_freq] * socket_num
        upper_uncore_freq = [max_uncore_freq] * socket_num

        # strategy_id == 2: default uncore frequency and multiple core frequency
        if self.strategy_id == 2:
            lower_core_freq = [self.arg] * core_num
            upper_core_freq = [self.arg] * core_num
            hardware_setter.set_core_frequency(lower_core_freq, upper_core_freq)
            print(f"[INFO] {now}ms,", f"current power {round(cur_power, 2)}w")
            log['time'] = now
            return log, pd.DataFrame({'core_freq': upper_core_freq})

        # strategy_id == 1: default core frequency and multiple uncore frequency
        if self.strategy_id == 1:
            lower_uncore_freq = [self.arg] * socket_num
            upper_uncore_freq = [self.arg] * socket_num
            hardware_setter.set_uncore_frequency(lower_uncore_freq, upper_uncore_freq)
            print(f"[INFO] {now}ms,", f"current power {round(cur_power, 2)}w")
            log['time'] = now
            return log, pd.DataFrame({'core_freq': upper_core_freq})

        # obtain buzy cores
        buzy_core = (cpu_util[cpu_util >= cpu_util_s].index).tolist()
        idle_core = (cpu_util[cpu_util < cpu_util_s].index).tolist()

        buzy_core_num_per_socket = [0] * socket_num
        for core_id in buzy_core:
            buzy_core_num_per_socket[core_id // core_num_per_socket] += 1        

        # node-level frequeny tuning
        if fc_strategy == "UCUFS" and self.strategy_id == 0:
            if len(buzy_core) != 0:
                mask = pred_data['pred_perf'] >= self.arg
                if torch.nonzero(mask).size(0) == 0:
                    for core_id in buzy_core:
                        lower_core_freq[core_id] = max_core_freq
                        socket_id = core_id // core_num_per_socket
                        lower_uncore_freq[socket_id] = max_uncore_freq
                else:
                    pred_energy_tmp = torch.where(mask, pred_data['pred_energy'], torch.full_like(pred_data['pred_energy'], 10000))
                    min_values, min_indices = torch.min(pred_energy_tmp, dim=1)
                    min_indices = min_indices.tolist()
                    lower_core_freq[:] = [freqs[min_indices[0], 0]] * core_num
                    upper_core_freq[:] = [freqs[min_indices[0], 0]]* core_num
                    lower_uncore_freq[:] = [freqs[min_indices[0], 1]]* socket_num
                    upper_uncore_freq[:] = [freqs[min_indices[0], 1]]* socket_num
                    
            print(f"[INFO] {now}ms, Set core0 to {lower_core_freq[0]}-{upper_core_freq[0]} MHz,", f"uncore to {lower_uncore_freq[0]}-{upper_uncore_freq[0]}, {lower_uncore_freq[1]}-{upper_uncore_freq[1]}")
            hardware_setter.set_frequency(lower_core_freq, upper_core_freq, lower_uncore_freq, upper_uncore_freq)
            log['time'] = now
            log['core'] = lower_core_freq
            log['uncore'] = lower_uncore_freq

            return log, pd.DataFrame({'core_freq': upper_core_freq})

        # core-level frequeny tuning
        if self.strategy_id == 0:
            mask = pred_data['pred_perf'] >= self.arg

            if len(buzy_core) != 0:
                if torch.nonzero(mask).size(0) == 0:
                    for core_id in buzy_core:
                        lower_core_freq[core_id] = max_core_freq
                        socket_id = core_id // core_num_per_socket
                        lower_uncore_freq[socket_id] = max_uncore_freq
                else:
                    pred_energy_tmp = torch.where(mask, pred_data['pred_energy'], torch.full_like(pred_data['pred_energy'], 10000))
                    min_values, min_indices = torch.min(pred_energy_tmp, dim=1)
                    min_indices = min_indices.tolist()

                    # Step1. search the optimal oncore frequency for each core
                    core_freq_id = [0] * core_num
                    for core_id in buzy_core:
                        core_freq_id[core_id] = min_indices[core_id] // uncore_freq_num
                        lower_core_freq[core_id] = freqs[min_indices[core_id], 0]
                        upper_core_freq[core_id] = freqs[min_indices[core_id], 0]

                    # Step2. search the optimial uncore frequency for each socket
                    for socket_id in range(socket_num):
                        if buzy_core_num_per_socket[socket_id] == 0:
                            continue
                        min_energy = 1000
                        min_perf = -1
                        opt_uncore_freq = max_uncore_freq
                        opt_uncore_freq_id = 0
                        for uncore_freq_id in range(uncore_freq_num):
                            uncore_freq = freqs[:uncore_freq_num, 1].tolist()[uncore_freq_id]
                            perf = 0
                            energy = 0
                            for core_id in buzy_core:
                                if core_id // core_num_per_socket == socket_id:
                                    idx = core_freq_id[core_id] * uncore_freq_num + uncore_freq_id
                                    perf += pred_data['pred_perf'][core_id][idx].item()
                                    energy += pred_data['pred_energy'][core_id][idx].item()
                            
                            # calculate averge performance and energy consumption
                            perf = perf / buzy_core_num_per_socket[socket_id]
                            energy = energy / buzy_core_num_per_socket[socket_id]
                            
                            if perf >= self.arg and energy < min_energy:
                                min_energy = energy
                                min_perf = perf
                                opt_uncore_freq = uncore_freq
                                opt_uncore_freq_id = uncore_freq_id
                        
                        lower_uncore_freq[socket_id] = opt_uncore_freq
                        upper_uncore_freq[socket_id] = opt_uncore_freq

                        # Step3. recorrect optimal oncore frequencies
                        # idx = list(range(opt_uncore_freq_id, opt_uncore_freq_id + uncore_freq_num * core_freq_num, uncore_freq_num))
                        # min_values, min_indices = pred_energy_tmp[:, idx].min(dim=1)
                        # original_cols_index = torch.tensor(idx)[min_indices].tolist()
                        # for core_id in buzy_core:
                        #     if core_id // core_num_per_socket == socket_id:
                        #         lower_core_freq[core_id] = freqs[original_cols_index[core_id], 0]
                        #         upper_core_freq[core_id] = freqs[original_cols_index[core_id], 0]

            print(f"[INFO] {now}ms, Set core0 to {lower_core_freq[0]}-{upper_core_freq[0]} MHz,", f"uncore to {lower_uncore_freq[0]}-{upper_uncore_freq[0]}, {lower_uncore_freq[1]}-{upper_uncore_freq[1]}")
            hardware_setter.set_frequency(lower_core_freq, upper_core_freq, lower_uncore_freq, upper_uncore_freq)
            log['time'] = now
            log['core'] = lower_core_freq
            log['uncore'] = lower_uncore_freq

            return log, pd.DataFrame({'core_freq': upper_core_freq})
