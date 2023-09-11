#!/bin/bash

#SBATCH --job-name="¯\\_(ツ)_/¯"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=40
#SBATCH --mem=600000
#SBATCH --time=5-00:00:00

srun python -m haplo.nicer_parameters_to_phase_amplitudes_train
