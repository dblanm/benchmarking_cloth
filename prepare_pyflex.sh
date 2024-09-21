#!/bin/bash
# Specify as input which GPU you want to use
echo "Setting CUDA_VISIBLE_DEVICES=$1"
export CUDA_VISIBLE_DEVICES=$1

ANACONDA_PATH=$HOME/anaconda3
SOFTGYM_PATH=${PWD}/deps/softgym
export PATH=${ANACONDA_PATH}/envs/pytorch3d/bin:$PATH
. activate pytorch3d
export PYFLEXROOT=${SOFTGYM_PATH}/PyFlex
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
