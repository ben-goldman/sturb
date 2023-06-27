#!/bin/sh
#
#
#SBATCH --account=thea        # Replace ACCOUNT with your group account name
#SBATCH --job-name=plots   # The job name
#SBATCH -c 2                     # The number of cpu cores to use (up to 32 cores per server)
#SBATCH -N 1
#SBATCH --time=0-0:30            # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=5G         # The memory the job will use per cpu core
 
module load anaconda
# conda init bash
source /burg/home/bog2101/.bashrc
conda activate spectralDNS
date
python go.py
ffmpeg -framerate 30 -pattern_type glob -i 'u2b2*.png' -c:v libx264 -pix_fmt yuv420p u2b2.mp4
date

# End of script
