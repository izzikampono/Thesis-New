#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --time=48:00:00
#SBATCH --error=error_file_Mars_densities.txt
#SBATCH --job-name=Mars
#SBATCH --mem=30G
#SBATCH --output=output_Mars_densities.log

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
cplex -c set parallel -1
cplex quit
cplex -c set threads 0
cplex quit
echo : "\n\n\n Loaded Cplex and set to parallel computing \n\n\n"

echo "Run problem : Mars with horizon: $1 and iter : $2, starting density : $3"
cd /scratch/s3918343/venvs/thesis/Thesis-New
python -m pip install joblib
python densities_experiment.py problem=Mars horizon=$1 iter=$2 density=$3
echo " SOLVING DONE"

deactivate
