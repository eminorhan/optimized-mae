#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_optimized_mae
#SBATCH --output=train_optimized_mae_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

srun python -u ../train_wds.py \
	--model 'mae_vit_huge_patch14' \
	--resume '' \
	--accum_iter 1 \
	--batch_size_per_gpu 256 \
	--input_size 224 \
	--mask_ratio 0.8 \
	--lr 0.0001 \
	--weight_decay 0.0 \
	--num_workers 16 \
	--output_dir /scratch/eo41/optimized-mae/outputs/sfp \
	--data_path "/scratch/eo41/data/saycam/Sfp_5fps_300s_{000000..000003}.tar" \
	--save_prefix sfp_vith14 \
	--compile

echo "Done"