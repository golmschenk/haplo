#!/bin/bash

#SBATCH --job-name="¯\\_(ツ)_/¯"
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=40
#SBATCH --mem=600000
#SBATCH --time=5-00:00:00

session_name="session_name"
current_time=$(date "+%Y_%m_%d_%H_%M_%S")
export HAPLO_SESSION_DIRECTORY="sessions/${current_time}_${session_name}"
mkdir -p "${HAPLO_SESSION_DIRECTORY}"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

srun torchrun \
--nnodes 4 \
--nproc_per_node gpu \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "$head_node_ip":36484 \
scripts/example_train_session.py
