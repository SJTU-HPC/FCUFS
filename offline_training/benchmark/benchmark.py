import os
import signal
import pandas as pd
import threading
import time
from utils import utils
from model import inference
from benchmark import gen_features
import subprocess

def obtain_benchmarks(directory):
    return [os.path.join(directory, entry) for entry in os.listdir(directory)]

def remove_prefix(path):
    filename_with_extension = os.path.basename(path)
    return os.path.splitext(filename_with_extension)[0]

class data_obtainer(threading.Thread):
    def __init__(self, moniter_path, stop_event, time_slice):
        super().__init__()
        self.result = None
        self.stop_event = stop_event
        self.moniter_path = moniter_path
        self.time_slice = time_slice

    def run(self):
        collected_data = list()
        process = subprocess.Popen([self.moniter_path, str(self.time_slice)], stdout=subprocess.PIPE)
        while not self.stop_event.is_set():
            line = process.stdout.readline()
            if not line:
                break
            collected_data.append(line.decode())
        process.terminate()

        self.result = collected_data

def run_benchmark(benchmark_path, moniter_path, time_slice):
    # start moniter
    stop_event = threading.Event()
    ob_thread = data_obtainer(moniter_path, stop_event, time_slice * 1000000)
    ob_thread.start()

    time.sleep(1)

    # start run benchmark
    process = subprocess.Popen(["/bin/sh", benchmark_path], stdout=subprocess.PIPE, preexec_fn=os.setsid)
    time.sleep(40)
    os.killpg(process.pid, signal.SIGTERM)

    # finished obtainer thread
    stop_event.set()
    time.sleep(1)
    ob_thread.join()

    return ob_thread.result
