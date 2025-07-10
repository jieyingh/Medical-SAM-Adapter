#!/bin/bash
#SBATCH --time=01:00:00 
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --account=def-banire
#SBATCH --job-name=MedSAM
#SBATCH --output=/lustre07/scratch/jieying/Medical-SAM-Adapter/out/02-MedSAM.out

# Load modules
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.10.0 

# Activate virtual environment
source ~/envs/sam-adapt/bin/activate

# Unzip data in job
mkdir $SLURM_TMPDIR/data
tar xf /lustre07/scratch/jieying/Medical-SAM-Adapter/data/train.tar -C $SLURM_TMPDIR/data

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 02-MedSAM-ZP \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/medsam/medsam_vit_b.pth \
    -d $SLURM_TMPDIR/data/train \
    -l ZP

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 02-MedSAM-PVS \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/medsam/medsam_vit_b.pth \
    -d $SLURM_TMPDIR/data/train \
    -l PVS

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 02-MedSAM-OO \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/medsam/medsam_vit_b.pth \
    -d $SLURM_TMPDIR/data/train \
    -l OO

python /lustre07/scratch/jieying/Medical-SAM-Adapter/train.py \
    -exp_name 02-MedSAM-PB \
    -vis 10 \
    -val_freq 1 \
    -b 5 -dataset oo \
    -sam_ckpt /lustre07/scratch/jieying/Medical-SAM-Adapter/checkpoint/medsam/medsam_vit_b.pth \
    -d $SLURM_TMPDIR/data/train \
    -l PB