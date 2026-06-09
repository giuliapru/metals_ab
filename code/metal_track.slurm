#!/bin/bash
#SBATCH --job-name=met_tracks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         
#SBATCH --time=12:00:00       
#SBATCH --output=/home/gpruto/metal_ab/code/metal_tracks.out
#SBATCH --error=/home/gpruto/metal_ab/code/metal_tracks.err
#SBATCH --mem=100GB     # Memory per node

source /etc/profile.d/modules.sh
module load anaconda/3.13
. /usr/local/anaconda/3.13/etc/profile.d/conda.sh

python /home/gpruto/metal_ab/code/metal_tracks.py
