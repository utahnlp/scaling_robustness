#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --mem=480G
#SBATCH --gres=gpu:a100:8

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=t5-11b_qnli_1
#SBATCH --output=final_models//qnli/t5-11b/seed-1/slurm_output.txt
#SBATCH --error=final_models//qnli/t5-11b/seed-1/slurm_output.txt
#SBATCH --time=3-00:00:00
#SBATCH --qos=soc-gpulong-np
#SBATCH --requeue




WORK_DIR=$HOME/scr/robustness/t5_robustness
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
#source env_t5_textattack_deepspeed//bin/activate
source envs/py385_ds_2/bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"





echo "deepspeed t5_run_glue_trainer.py --deepspeed ds_config_zero3.json --model_name_or_path t5-11b --per_device_train_batch_size 4 --num_train_epochs 3 --seed 1 --task_name qnli --do_eval --do_train --per_device_eval_batch_size 4 --overwrite_output_dir --output_dir final_models/qnli/t5-11b/seed-1/ --save_steps 1000000 --learning_rate 5e-6 --bf16 --predict_with_generate --max_source_length 256 | tee final_models/qnli/t5-11b/seed-1/train_output_log.txt"


deepspeed t5_run_glue_trainer.py --deepspeed ds_config_zero3.json --model_name_or_path t5-11b --per_device_train_batch_size 8 --num_train_epochs 3 --seed 1 --task_name qnli --do_eval --do_train --per_device_eval_batch_size 8 --overwrite_output_dir --output_dir final_models/qnli/t5-11b/seed-1/ --save_steps 1000000 --learning_rate 5e-6 --bf16 --predict_with_generate --max_source_length 256 | tee final_models/qnli/t5-11b/seed-1/train_output_log.txt
