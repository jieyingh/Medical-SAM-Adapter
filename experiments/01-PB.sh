#!/bin/bash
#SBATCH --time=08:00:00 
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --account=def-banire
#SBATCH --job-name=SAM-ZP
#SBATCH --output=/lustre07/scratch/jieying/Medical-SAM-Adapter/out/01-PB.out

# Load modules
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.10.0 

# Activate virtual environment
source ~/envs/sam-adapt/bin/activate

# Unzip data in job
mkdir $SLURM_TMPDIR/data
tar xf /lustre07/scratch/jieying/Medical-SAM-Adapter/data/train.tar -C $SLURM_TMPDIR/data

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 01-ZP \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/sam/sam_vit_b_01ec64.pth \
    -d $SLURM_TMPDIR/data/train \
    -l PB