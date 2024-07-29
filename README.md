# FCUFS - Fine-grained Core and Uncore Frequency Scaling

## Introduction

FCUFS is an open source framework for saving energy consumption with controllable performance loss. FCUFS operates at fixed intervals, sensing performance characteristics, predicting performance and power for the next timeframe, and adjusting frequencies based on prior predictions. It is transparent to workloads, tuning both oncore and uncore frequencies at the core level, thereby reducing energy consumption while controlling performance loss.

## Dependency

Ensure you have Python 3.8. 

Install the required Python dependencies using pip:

```shell
pip install -r requirements.txt
```

## Usage

Make sure you have root permissions. Disable intel_pstate and set power mode to "ondemand" governor with the following commands.

```shell
echo passive | sudo tee /sys/devices/system/cpu/intel_pstate/status
cpupower frequency-set -g ondemand
```

### Offline training

Enter the offline training directory and compile the sampler.

```shell
cd offline_training
./build.sh
```

#### Step 1. Sampling performance events

Modify the benchmark paths, frequency steps, node information and other configurations in lines 14-34 of collect_data.py according to the comments.

Run the data sampling script:

```shell
OMP_NUM_THREADS=1 python collect_data.py
```

#### Step 2. Processing data

```shell
python gen_features.py && python gen_dataset.py
```

#### Step 3. Training

Train power and performance prediction models:

```shell
python train_mlp.py power
python train_mlp.py performance
```

Copy trained models to online_tuning directory:

```shell
cp ./*CUFS.pth ../online_tuning/trained_model/ 
```

### Online tuning

Enter the online tuning directory and compile the sampler.

```shell
cd ../online_tuning
./build.sh
```

#### Launch online tuning service

Modify configuration file "config.json".

Launch online tuning service:

```shell
OMP_NUM_THREADS=1 python online_tuning.py config.json
```