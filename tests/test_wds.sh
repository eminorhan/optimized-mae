#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=00:10:00
#SBATCH --job-name=test_wds
#SBATCH --output=test_wds_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

srun python -u test_wds.py \
	--batch_size_per_gpu 256 \
	--input_size 224 \
	--num_workers 16 \
	--data_path "/scratch/eo41/data/saycam/Sfp_5fps_300s_{000000..000003}.tar"

echo "Done"