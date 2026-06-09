#!/bin/bash
#SBATCH --job-name=g37591_all_5_cold_gas_only        # Job name
#SBATCH --output=/home/gpruto/metal_ab/code/jobs/all_gas/cold_gas_only/g37591_all_z5_coldgas.out       # Output file
#SBATCH --error=/home/gpruto/metal_ab/code/jobs/all_gas/cold_gas_only/g37591_all_z5_coldgas.err        # error file
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
python /home/gpruto/metal_ab/code/metals_singlegal_total.py 92 True g37591
