#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=128GB
#SBATCH --requeue
#SBATCH --job-name=big_job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ab10945@nyu.edu

conda activate my_env
python3 main.py --model XLM-T --dataset UMSAB --task inference --cpt_dir checkpoint_logs --op_dir output_logs 
