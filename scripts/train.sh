#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --job-name=train_optimized_mae
#SBATCH --output=train_optimized_mae_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

NUM_IGS=10000

srun python -u ../train.py \
	--model 'mae_vit_huge_patch14_448' \
	--resume '' \
	--accum_iter 1 \
	--batch_size_per_gpu 128 \
	--input_size 448 \
	--mask_ratio 0.8 \
	--lr 0.0001 \
	--min_lr 0.0001 \
	--weight_decay 0.0 \
	--num_workers 16 \
	--output_dir /scratch/eo41/optimized-mae/outputs/models_sa1b_${NUM_IGS} \
	--data_path /vast/eo41/sa-1b/images_${NUM_IGS}/${SLURM_ARRAY_TASK_ID} \
	--save_prefix vith14_448_${NUM_IGS}_${SLURM_ARRAY_TASK_ID} \
	--compile

echo "Done"