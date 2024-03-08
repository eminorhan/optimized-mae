#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --job-name=finetune_optimized_mae
#SBATCH --output=finetune_optimized_mae_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

NUM_IGS=10000

srun python -u ../finetune.py \
	--model vit_huge_patch14_1792 \
	--resume /scratch/eo41/optimized-mae/outputs/models_sa1b_${NUM_IGS}/vith14_1792_${NUM_IGS}_${SLURM_ARRAY_TASK_ID}_checkpoint.pth \
	--save_prefix vith14_1792_${NUM_IGS}_${SLURM_ARRAY_TASK_ID}_finetuned \
	--input_size 1792 \
	--batch_size_per_gpu 3 \
	--accum_iter 4 \
	--epochs 50 \
	--num_workers 16 \
	--lr 0.0001 \
	--output_dir /scratch/eo41/optimized-mae/outputs/models_sa1b_${NUM_IGS}_finetuned \
	--train_data_path /scratch/work/public/imagenet/train \
	--val_data_path /scratch/eo41/imagenet/val \
	--num_labels 1000 \
	--compile

echo "Done"
