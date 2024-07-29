#!/bin/sh

ROOT_PATH=`dirname "${BASH_SOURCE[0]}"`
ROOT_PATH=`realpath $ROOT_PATH`

time OMP_PROC_BIND=true OMP_NUM_THREADS=20 taskset -c 0-19 $ROOT_PATH/ep_20.x &

time OMP_PROC_BIND=true OMP_NUM_THREADS=10 taskset -c 20-29 $ROOT_PATH/lu_10.x &

time OMP_PROC_BIND=true OMP_NUM_THREADS=10 taskset -c 30-39 $ROOT_PATH/bt_10.x &

wait
