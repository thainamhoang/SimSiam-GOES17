#!/bin/bash

# example using singularity

# you need permission to submit jobs to the salvador partition.
# alternately, use "all" partition for testing, this will work due to
# the 'gres=gpu:1'. Salvador partition jobs can preempt "All" partition

#SBATCH --gpus=1 # 1 gpu
#SBATCH --time=00:20:00 # 20 minute wall clock limit
#SBATCH --output=/home/%u/output/sb_%j.log
#output directory must exist or the job will silently fail

#NOTE - cpu-per-gpu and mem-per-gpu defaults are 16G of ram per GPU and 6 cores.
# these are reasonable defaults, but can be changed with --cpus-per-gpu and --mem-per-gpu

#point at shared container
CONTAINER=/home/shared/containers/pytorch_22.04-py3.sif

source /etc/profile

# Run script
singularity run --nv $CONTAINER python /$HOME/gpu/SimSiam-GOES17/main_simsiam.py