#!/bin/bash
#SBATCH --time=01:00:00 
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --account=def-banire
#SBATCH --job-name=tar-results
#SBATCH --output=/lustre07/scratch/jieying/Medical-SAM-Adapter/out/tar-results.out

tar -czf Results.tar.gz -C /lustre07/scratch/jieying/Medical-SAM-Adapter/experiments .