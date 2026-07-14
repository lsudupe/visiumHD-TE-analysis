#!/bin/bash
#SBATCH -p batch
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH -J build_cell
#SBATCH -o log.%J.out
#SBATCH -e err.%J.err
#SBATCH --cpus-per-task=4

cd /ibex/user/medinils/visiumHD-TE-analysis
source ~/.bashrc
conda activate visiumhd

SAMPLE=$1
TE_LEVEL=${2:-family}
python scripts/build_cell_object.py --sample "${SAMPLE}" --te-level "${TE_LEVEL}"

