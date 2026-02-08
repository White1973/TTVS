#!/bin/bash
#SBATCH --job-name=bsk_ttrl          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=4                 # number of GPUs per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal  # partition(queue) where you submit
#SBATCH --account=peilab      # only require for multiple projects

module purge                     # clear environment modules inherited from submission
module load Anaconda3/2023.09-0  # load the exact modules required
source activate ttrl
ray stop
ray start --head
export RAY_ADDRESS="auto"
export RAY_TMPDIR="/project/peilab/bsk/ray_tmpdir1"
bash examples/ttrl/Qwen2.5-Instruct/math_16.sh