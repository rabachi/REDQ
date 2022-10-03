#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH -p 'rtx6000,t4v1,t4v2,p100'           # partition
#SBATCH --cpus-per-task=1     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude="gpu022,gpu023,gpu024,gpu025"
#SBATCH --time=20:00:00      # time limit
#SBATCH --mem=8GB         # minimum amount of real memory
#SBATCH --job-name=redq

source ~/.bashrc
export MUJOCO_PY_BYPASS_LOCK=True
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/abachiro/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

conda activate viper
module load cuda-11.3

cd ~/REDQ

hostname
nvidia-smi

MUJOCO_GL=egl python experiments/train_redq_sac.py \
    --env_type dmc \
    --env walker-walk-v99 \
    --seed $RANDOM \
    --data_dir /checkpoint/abachiro/viper_results/$SLURM_JOB_ID
