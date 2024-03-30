#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=roberta-base_qqp_1
#SBATCH --output=final_models//qqp/roberta-base/seed-1/slurm_output.txt
#SBATCH --error=final_models//qqp/roberta-base/seed-1/slurm_output.txt
#SBATCH --time=12:00:00




WORK_DIR=$HOME/scr/robustness/t5_robustness
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
source env_t5_textattack//bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"



echo "python run_glue.py --model_name_or_path roberta-base --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --num_train_epochs 3 --seed 1 --task_name qqp --do_train --do_eval --save_steps 1000000 --max_seq_length 256 --overwrite_output_dir --output_dir final_models//qqp/roberta-base/seed-1 --learning_rate 5e-05"



python run_glue.py --model_name_or_path roberta-base --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --num_train_epochs 3 --seed 1 --task_name qqp --do_train --do_eval --save_steps 1000000 --max_seq_length 256 --overwrite_output_dir --output_dir final_models//qqp/roberta-base/seed-1 --learning_rate 5e-05



