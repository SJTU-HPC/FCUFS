#!/bin/sh

ROOT_PATH=`dirname "${BASH_SOURCE[0]}"`
ROOT_PATH=`realpath $ROOT_PATH`
export OMP_PROC_BIND=true
OMP_NUM_THREADS=40 $ROOT_PATH/bt.D.x
