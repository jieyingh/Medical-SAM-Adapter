#!/bin/bash
#SBATCH --time=05:00:00 
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --account=def-banire
#SBATCH --job-name=MedSAMBox
#SBATCH --output=/lustre07/scratch/jieying/Medical-SAM-Adapter/out/07-VanillaFullPt.out

# Load modules
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.10.0 

# Activate virtual environment
source ~/envs/sam-adapt/bin/activate

# Unzip data in job
mkdir $SLURM_TMPDIR/data
tar xf /lustre07/scratch/jieying/Medical-SAM-Adapter/data/train_full.tar -C $SLURM_TMPDIR/data

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 07-VanillaFullPt-ZP \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/sam/sam_vit_b_01ec64.pth \
    -d $SLURM_TMPDIR/data/train \
    -l ZP

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 07-VanillaFullPt-PVS \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/sam/sam_vit_b_01ec64.pth \
    -d $SLURM_TMPDIR/data/train \
    -l PVS

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 07-VanillaFullPt-OO \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/sam/sam_vit_b_01ec64.pth \
    -d $SLURM_TMPDIR/data/train \
    -l OO

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 07-VanillaFullPt-PB \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/sam/sam_vit_b_01ec64.pth \
    -d $SLURM_TMPDIR/data/train \
    -l PB