#!/bin/bash
#SBATCH -p batch
#SBATCH --time=04:00:00
#SBATCH --mem=200GB
#SBATCH -J build_spatial
#SBATCH -o log.%J.out
#SBATCH -e err.%J.err
#SBATCH --cpus-per-task=8

cd /ibex/user/medinils/visiumHD-TE-analysis

source ~/.bashrc
conda activate visiumhd

# usage: sbatch scripts/run_build_spatial_object.sh <sample> <te_level>
SAMPLE=$1
TE_LEVEL=${2:-family}

python scripts/build_spatial_object.py --sample "${SAMPLE}" --te-level "${TE_LEVEL}"

