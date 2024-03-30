#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:4

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=t5-base_pietrolesci/nli_fever_3
#SBATCH --output=final_models//pietrolesci/nli_fever/t5-base/seed-3/slurm_output.txt
#SBATCH --error=final_models//pietrolesci/nli_fever/t5-base/seed-3/slurm_output.txt
#SBATCH --time=12:00:00




WORK_DIR=$HOME/scr/robustness/t5_robustness
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
source env_t5_textattack//bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"



echo "python t5_run_glue_trainer.py --model_name_or_path t5-base --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 3 --seed 3 --task_name pietrolesci/nli_fever --do_train --save_steps 1000000 --max_source_length 512 --overwrite_output_dir --output_dir final_models//pietrolesci/nli_fever/t5-base/seed-3 --learning_rate 0.0001 --predict_with_generate"



python t5_run_glue_trainer.py --model_name_or_path t5-base --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 3 --seed 3 --task_name pietrolesci/nli_fever --do_train --save_steps 1000000 --max_source_length 512 --overwrite_output_dir --output_dir final_models//pietrolesci/nli_fever/t5-base/seed-3 --learning_rate 0.0001 --predict_with_generate



