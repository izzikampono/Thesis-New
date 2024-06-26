#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120
#SBATCH --time=24:00:00
#SBATCH --error=error_file_wirelessDelay_small.txt
#SBATCH --job-name=wirelessDelay_small
#SBATCH --mem=70G
#SBATCH --output=output_wirelessDelay_small.log

module purge
module load Python/3.9.6-GCCcore-11.2.0


# Check if at least one argument is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <problem> [<horizon>] [num_iter]"
    exit 1
fi
source /scratch/s3918343/venvs/thesis/bin/activate
echo : "initialized python evironment"
module load CPLEX/22.1.1-GCCcore-11.2.0


echo : "\n\n\n Loaded Cplex and set to parallel computing \n\n\n"
cd /scratch/s3918343/venvs/thesis/Thesis-New
echo "Run problem : wirelessDelay with horizon: $1 and iter : $2"
python experiment_script.py problem=wirelessDelay horizon=$1 iter=$2 density=0.1
echo " SOLVING DONE"

deactivate
