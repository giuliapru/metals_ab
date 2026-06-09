#!/bin/bash
#SBATCH --job-name=g39_metals
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         
#SBATCH --time=12:00:00       
#SBATCH --output=/home/gpruto/CGM_ref_analysis/code/metals/g39_metals.out
#SBATCH --error=/home/gpruto/CGM_ref_analysis/code/metals/g39_metals.err
#SBATCH --mem=100GB     # Memory per node

source /etc/profile.d/modules.sh
module load anaconda/3.13
. /usr/local/anaconda/3.13/etc/profile.d/conda.sh

python /home/gpruto/CGM_ref_analysis/code/metals/metals_singlegal_total.py 68 False
