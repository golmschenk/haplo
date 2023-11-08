#!/bin/bash

#SBATCH --job-name="¯\\_(ツ)_/¯"
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=40
#SBATCH --mem=600000
#SBATCH --time=5-00:00:00

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
--nnodes 4 \
--nproc_per_node gpu \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:36484 \
-m haplo.nicer_parameters_to_phase_amplitudes_train
