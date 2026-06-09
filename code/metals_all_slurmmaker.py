import os
import subprocess
import time
import numpy as np


snaps = np.array([51, 68, 92]) #39 51, 68, 92
redshifts = np.array([7,6,5]) #8,7,6,5
cold_gas_only = False
gal = ['g578', 'g205', 'g39']#['g5229300', 'g2274036', 'g519761', 'g500531', 'g137030', 'g37591','g33206', 'g10304', 'g5760', 'g1163', 'g578', 'g205', 'g39']


def create_slurm_script(job_name, output_file, error_file, times, nodes, ntasks_per_node, job_script_name, job_commands):
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}        # Job name
#SBATCH --output={output_file}       # Output file
#SBATCH --error={error_file}        # error file
#SBATCH -p all
#SBATCH --time={times}                # Time limit
#SBATCH --nodes={nodes}              # Number of nodes
#SBATCH --mem-per-cpu=100GB
#SBATCH --ntasks-per-node={ntasks_per_node}  # Number of tasks per node

# Load any necessary modules (if needed)
source /etc/profile.d/modules.sh
module load anaconda/3.13
. /usr/local/anaconda/3.13/etc/profile.d/conda.sh

# Insert commands to run the job
{job_commands}
"""
        # Write the SLURM script to a file
    with open(job_script_name, 'w') as f:
        f.write(slurm_script)

    
# Function to submit SLURM job using sbatch
def submit_slurm_job(job_script_name):
    try:
        result = subprocess.run(['sbatch', job_script_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Job submitted successfully: {result.stdout.strip()}")
        else:
            print(f"Error in job submission: {result.stderr.strip()}")
    except Exception as e:
        print(f"Failed to submit job: {e}")

for g in gal:
    for i in range(len(snaps)):
        # Example parameters
        job_name = "%s_all_%d" %(g, redshifts[i])
        output_file = "/home/gpruto/metal_ab/code/jobs/all_gas/%s_all_z%d.out" %(g, redshifts[i]) # %j will be replaced with the job ID
        error_file = "/home/gpruto/metal_ab/code/jobs/all_gas/%s_all_z%d.err" %(g, redshifts[i])
        times = "05:00:00"  
        nodes = 1
        ntasks_per_node = 1
        job_script_name = "/home/gpruto/metal_ab/code/jobs/all_gas/%s_all_z%d.slurm" %(g, redshifts[i])
        job_commands = "python /home/gpruto/metal_ab/code/metals_singlegal_total.py %d %s %s" %(snaps[i], cold_gas_only, g) 

        if cold_gas_only==True:
            job_name += "_cold_gas_only"
            output_file = "/home/gpruto/metal_ab/code/jobs/all_gas/%s_all_z%d_coldgas.out" %(g, redshifts[i]) # %j will be replaced with the job ID
            error_file = "/home/gpruto/metal_ab/code/jobs/all_gas/%s_all_z%d_coldgas.err" %(g, redshifts[i])
            job_script_name = "/home/gpruto/metal_ab/code/jobs/all_gas/%s_all_z%d_coldgas.slurm" %(g, redshifts[i])
            job_commands = "python /home/gpruto/metal_ab/code/metals_singlegal_total.py %d %s %s" %(snaps[i], cold_gas_only, g)

        # Create the SLURM script
        create_slurm_script(job_name, output_file, error_file, times, nodes, ntasks_per_node, job_script_name, job_commands)

        # Submit the SLURM job
        submit_slurm_job(job_script_name)

