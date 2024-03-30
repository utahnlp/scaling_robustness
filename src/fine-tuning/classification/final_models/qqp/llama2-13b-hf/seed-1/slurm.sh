#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --mem=0
#SBATCH --gres=gpu:a100:8

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=llama2-13b-hf_qqp_1
#SBATCH --output=/scratch/general/vast/u1266434/robustness/t5_robustness/llama2_ft_cls/final_models//qqp/llama2-13b-hf/seed-1/slurm_output.txt
#SBATCH --error=/scratch/general/vast/u1266434/robustness/t5_robustness/llama2_ft_cls/final_models//qqp/llama2-13b-hf/seed-1/slurm_output.txt
#SBATCH --time=3-00:00:00
#SBATCH --qos=soc-gpulong-np
#SBATCH --requeue


WORK_DIR=$HOME/scr/robustness/t5_robustness/llama2_ft_cls/
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
#source env_t5_textattack_deepspeed/bin/activate
source ../envs/py3104_transformers/bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"



echo "deepspeed run_glue.py --deepspeed ds_config_zero3.json --model_name_or_path llama2-13b-hf --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 3 --seed 1 --task_name qqp --do_train --do_eval --save_steps 1000000 --max_seq_length 256 --overwrite_output_dir --output_dir final_models//qqp/llama2-13b-hf/seed-1 --learning_rate 1e-5 --bf16"

deepspeed run_glue.py --deepspeed ds_config_zero3_alt.json --model_name_or_path llama-2-13b-hf --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 3 --seed 1 --task_name qqp --do_train --do_eval --save_steps 1000000 --max_seq_length 256 --overwrite_output_dir --output_dir final_models//qqp/llama2-13b-hf/seed-1 --learning_rate 1e-5 --bf16 | tee final_models//qqp/llama2-13b-hf/seed-1/train_output_log.txt

