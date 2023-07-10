#!/usr/bin/env bash
#SBATCH -t 1-00:00:00
#SBATCH --gpus-per-node=p100  
#SBATCH --output=outputs/slurm-%A_%a.out 
#SBATCH --mem=20000
#SBATCH --account=rrg-rgmelko-ab
#SBATCH --mail-user=msmoss@uwaterloo.ca
#SBATCH --mail-type=ALL

module load cuda/11.2.2 cudnn/8.2.0

nvidia-smi

export TF_GPU_ALLOCATOR=cuda_malloc_async

source ../ReconstructRydberg/bin/activate

echo $delta
echo $dim
echo $nh

python script_data_training.py \
    $delta 10000 \
    --rnn_dim $dim --nh $nh \
    --seed $seed \
    --qmc_data \
    --dset_size 1000 \

