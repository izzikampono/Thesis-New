#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=50:00:00
#SBATCH --error=error_file_mar_smalls.txt
#SBATCH --job-name=Mars_small
#SBATCH --mem=50G
#SBATCH --output=output_mars_small.log

module purge
module load Python/3.9.6-GCCcore-11.2.0


# Check if at least one argument is provided
if [ $# -lt 2]; then
    echo "Usage: $0 <problem> [<horizon>] [num_iter]"
    exit 1
fi
source /scratch/s3918343/venvs/thesis/bin/activate
echo : "initialized python evironment"
module load CPLEX/22.1.1-GCCcore-11.2.0
cplex -c set parallel -1
cplex quit
cplex -c set threads 0
cplex quit
echo : "\n\n\n Loaded Cplex and set to parallel computing \n\n\n"

echo : "\n\n\n Loaded Cplex and set to parallel computing \n\n\n"
cd /scratch/s3918343/venvs/thesis/Thesis-New
echo "Run problem : Mars with horizon: $1 and iter : $2"
python experiment_script.py problem=Mars horizon=$1 iter=$2 density=0.1
echo " SOLVING DONE"

deactivate
