#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --mem=0
#SBATCH --gres=gpu:a100:8

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=facebook_opt-13b_snli_1
#SBATCH --output=final_models//snli/facebook/opt-13b/seed-1/slurm_output.txt
#SBATCH --error=final_models//snli/facebook/opt-13b/seed-1/slurm_output.txt
#SBATCH --time=3-00:00:00
#SBATCH --qos=soc-gpulong-np
#SBATCH --requeue


WORK_DIR=$HOME/scr/robustness/t5_robustness
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
source env_t5_textattack_deepspeed/bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"




echo "deepspeed run_glue.py --deepspeed ds_config_zero3.json --model_name_or_path facebook/opt-13b --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 3 --seed 1 --task_name snli --do_train --do_eval --save_steps 1000000 --max_seq_length 256 --overwrite_output_dir --output_dir final_models//snli/facebook/opt-13b/seed-1 --learning_rate 1e-6 --bf16"

deepspeed run_glue.py --deepspeed ds_config_zero3.json --model_name_or_path facebook/opt-13b --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 3 --seed 1 --task_name snli --do_train --do_eval --save_steps 1000000 --max_seq_length 256 --overwrite_output_dir --output_dir final_models//snli/facebook/opt-13b/seed-1 --learning_rate 1e-6 --bf16 | tee final_models//snli/facebook/opt-13b/seed-1/train_output_log.txt



