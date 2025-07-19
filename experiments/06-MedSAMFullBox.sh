#!/bin/bash
#SBATCH --time=05:00:00 
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --account=def-banire
#SBATCH --job-name=MedSAMBox
#SBATCH --output=/lustre07/scratch/jieying/Medical-SAM-Adapter/out/06-MedSAMFullBox.out

# Load modules
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.10.0 

# Activate virtual environment
source ~/envs/sam-adapt/bin/activate

# Unzip data in job
mkdir $SLURM_TMPDIR/data
tar xf /lustre07/scratch/jieying/Medical-SAM-Adapter/data/train_full.tar -C $SLURM_TMPDIR/data

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 06-MedSAMFullBox-ZP \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/medsam/medsam_vit_b.pth \
    -d $SLURM_TMPDIR/data/train \
    -l ZP

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 06-MedSAMFullBox-PVS \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/medsam/medsam_vit_b.pth \
    -d $SLURM_TMPDIR/data/train \
    -l PVS

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 06-MedSAMFullBox-OO \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/medsam/medsam_vit_b.pth \
    -d $SLURM_TMPDIR/data/train \
    -l OO

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 06-MedSAMFullBox-PB \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/medsam/medsam_vit_b.pth \
    -d $SLURM_TMPDIR/data/train \
    -l PB