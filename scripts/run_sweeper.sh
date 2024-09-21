#!/bin/bash

dataset_folder=/mnt/hdd/david/datasets/Dynamic_cloth_bench/rgb_quasi_static_and_dynamic



SIM_ENV=softgym
primitive=dynamic
primitive=quasi_static
optimiser=gpyopt

python -m bcm --multirun envs=${SIM_ENV} target=chequered_rag_0 data_path=${dataset_folder} primitive=${primitive} hydra/sweeper/sampler=$optimiser
python -m bcm --multirun envs=${SIM_ENV} target=cotton_rag_0 data_path=${dataset_folder} primitive=${primitive} hydra/sweeper/sampler=$optimiser
python -m bcm --multirun envs=${SIM_ENV} target=linen_rag_0 data_path=${dataset_folder} primitive=${primitive} hydra/sweeper/sampler=$optimiser





python -m bcm --multirun envs=${SIM_ENV}  'target=white_rag_0' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/white_rag_0/cloud'  'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=white_rag_1' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/white_rag_1/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=white_rag_2' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/white_rag_2/cloud' 'using_multi_run=True'
python -m bcm --multirun envs=${SIM_ENV}  'target=towel_rag_0' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/towel_rag_0/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=towel_rag_1' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/towel_rag_1/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=towel_rag_2' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/towel_rag_2/cloud' 'using_multi_run=True'
python -m bcm --multirun envs=${SIM_ENV}  'target=chequered_rag_0' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/chequered_rag_0/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=chequered_rag_1' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/chequered_rag_1/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=chequered_rag_2' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/chequered_rag_2/cloud' 'using_multi_run=True'

#SIM_ENV=mujoco

#python -m bcm --multirun envs=${SIM_ENV}  'target=white_rag_0' 'target_path=/home/david/datasets/Dyn$
#python -m bcm --multirun envs=${SIM_ENV}  'target=white_rag_1' 'target_path=/home/david/datasets/Dyn$
#python -m bcm --multirun envs=${SIM_ENV}  'target=white_rag_2' 'target_path=/home/david/datasets/Dyn$
#python -m bcm --multirun envs=${SIM_ENV}  'target=towel_rag_0' 'target_path=/home/david/datasets/Dyn$
#python -m bcm --multirun envs=${SIM_ENV}  'target=towel_rag_1' 'target_path=/home/david/datasets/Dyn$
#python -m bcm --multirun envs=${SIM_ENV}  'target=towel_rag_2' 'target_path=/home/david/datasets/Dyn$
#python -m bcm --multirun envs=${SIM_ENV}  'target=chequered_rag_0' 'target_path=/home/david/datasets$
#python -m bcm --multirun envs=${SIM_ENV}  'target=chequered_rag_1' 'target_path=/home/david/datasets$
#python -m bcm --multirun envs=${SIM_ENV}  'target=chequered_rag_2' 'target_path=/home/david/datasets$





SIM_ENV=mujoco
python -m bcm --multirun envs=${SIM_ENV}  'target=white_rag_0' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/white_rag_0/cloud'  'using_multi_run=True'
python -m bcm --multirun envs=${SIM_ENV}  'target=towel_rag_0' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/towel_rag_0/cloud' 'using_multi_run=True'
python -m bcm --multirun envs=${SIM_ENV}  'target=chequered_rag_0' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/chequered_rag_0/cloud' 'using_multi_run=True'

SIM_ENV=bullet
python -m bcm --multirun envs=${SIM_ENV}  'target=white_rag_0' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/white_rag_0/cloud'  'using_multi_run=True'
python -m bcm --multirun envs=${SIM_ENV}  'target=towel_rag_0' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/towel_rag_0/cloud' 'using_multi_run=True'
python -m bcm --multirun envs=${SIM_ENV}  'target=chequered_rag_0' 'target_path=/home/david/datasets/Dynamic_cloth_bench/demos/chequered_rag_0/cloud' 'using_multi_run=True'

SIM_ENV=sofa

#SIM_ENV=sofa

#python -m bcm --multirun envs=${SIM_ENV}  'target=white_rag_0' 'target_path=/home/mulerod1/datasets/Dynamic_cloth_bench/demos/white_rag_0/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=white_rag_1' 'target_path=/home/mulerod1/datasets/Dynamic_cloth_bench/demos/white_rag_1/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=white_rag_2' 'target_path=/home/mulerod1/datasets/Dynamic_cloth_bench/demos/white_rag_2/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=towel_rag_0' 'target_path=/home/mulerod1/datasets/Dynamic_cloth_bench/demos/towel_rag_0/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=towel_rag_1' 'target_path=/home/mulerod1/datasets/Dynamic_cloth_bench/demos/towel_rag_1/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=towel_rag_2' 'target_path=/home/mulerod1/datasets/Dynamic_cloth_bench/demos/towel_rag_2/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=chequered_rag_0' 'target_path=/home/mulerod1/datasets/Dynamic_cloth_bench/demos/chequered_rag_0/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=chequered_rag_1' 'target_path=/home/mulerod1/datasets/Dynamic_cloth_bench/demos/chequered_rag_1/cloud' 'using_multi_run=True'
#python -m bcm --multirun envs=${SIM_ENV}  'target=chequered_rag_2' 'target_path=/home/mulerod1/datasets/Dynamic_cloth_bench/demos/chequered_rag_2/cloud' 'using_multi_run=True'

