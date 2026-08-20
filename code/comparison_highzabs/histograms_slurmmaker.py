import os
import subprocess
import time
import numpy as np
import sys

snaps = np.array([30, 39, 51])# 129, 188]) #30 39 51, 68, 92, 129, 188
redshifts = np.array([9,8,7]) #np.array([8, 7, 6, 5, 4, 3]) #8,7,6,5, 4
gal = ['g5229300', 'g2274036', 'g519761', 'g500531', 'g137030', 'g37591','g33206', 'g10304', 'g5760', 'g1163', 'g578', 'g205', 'g39', 'g2']
density_cut = -2

def create_slurm_script(job_name, output_file, error_file, times, nodes, mem_per_cpu,  ntasks_per_node, job_script_name, job_commands):
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}        # Job name
#SBATCH --output={output_file}       # Output file
#SBATCH --error={error_file}        # error file
#SBATCH -p all
#SBATCH --time={times}                # Time limit
#SBATCH --nodes={nodes}              # Number of nodes
#SBATCH --mem-per-cpu={mem_per_cpu}
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
        job_name = "highz_%s_z%d" %(g, redshifts[i])
        output_file = "/home/gpruto/metal_ab/code/jobs/highz_abs/highz_hist_%s_z%d.out" %(g, redshifts[i]) # %j will be replaced with the job ID
        error_file = "/home/gpruto/metal_ab/code/jobs/highz_abs/highz_hist_%s_z%d.err" %(g, redshifts[i])
        times = "05:00:00"  if g=='g2' else "01:00:00"
        nodes = 1
        mem_per_cpu = "100GB" if g == 'g39' or g=='g205' or g=='g578' else "200GB" if g=='g2' else "50GB"  # Adjust memory based on galaxy
        ntasks_per_node = 1
        job_script_name = "/home/gpruto/metal_ab/code/jobs/highz_abs/highz_hist_%s_z%d.slurm" %(g, redshifts[i])
        job_commands = "python /home/gpruto/metal_ab/code/comparison_highzabs/histograms.py %d %s %.1f" %(snaps[i], g, density_cut) 
    
        # Create the SLURM script
        create_slurm_script(job_name, output_file, error_file, times, nodes, mem_per_cpu, ntasks_per_node, job_script_name, job_commands)

        # Submit the SLURM job
        submit_slurm_job(job_script_name)

