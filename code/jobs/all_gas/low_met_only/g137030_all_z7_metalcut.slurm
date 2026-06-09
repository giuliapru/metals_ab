#!/bin/bash
#SBATCH --job-name=g137030_all_7_metal_cut        # Job name
#SBATCH --output=/home/gpruto/metal_ab/code/jobs/all_gas/low_met_only/g137030_all_z7_metalcut.out       # Output file
#SBATCH --error=/home/gpruto/metal_ab/code/jobs/all_gas/low_met_only/g137030_all_z7_metalcut.err        # error file
#SBATCH -p all
#SBATCH --time=05:00:00                # Time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --mem-per-cpu=10GB
#SBATCH --ntasks-per-node=1  # Number of tasks per node

# Load any necessary modules (if needed)
source /etc/profile.d/modules.sh
module load anaconda/3.13
. /usr/local/anaconda/3.13/etc/profile.d/conda.sh

# Insert commands to run the job
python /home/gpruto/metal_ab/code/metals_singlegal_total.py 51 False g137030 True
