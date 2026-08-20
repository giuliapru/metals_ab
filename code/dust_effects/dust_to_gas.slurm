#!/bin/bash
#SBATCH --job-name=dust_hist       # Job name
#SBATCH --output=/home/gpruto/metal_ab/code/dust_effects/dust_to_gas.out       # Output file
#SBATCH --error=/home/gpruto/metal_ab/code/dust_effects/dust_to_gas.err  # error file
#SBATCH -p all
#SBATCH --time=15:00:00                # Time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --mem-per-cpu=100GB
#SBATCH --ntasks-per-node=1  # Number of tasks per node

# Load any necessary modules (if needed)
source /etc/profile.d/modules.sh
module load anaconda/3.13
. /usr/local/anaconda/3.13/etc/profile.d/conda.sh

# Insert commands to run the job
python /home/gpruto/metal_ab/code/dust_effects/dust_to_gas.py
