#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --mem=480G
#SBATCH --gres=gpu:a100:8
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=boolq
#SBATCH --output=/scratch/general/vast/u1266434/robustness/t5_robustness/final_models/boolq/t5-11b/seed-1/slurm_output.txt
#SBATCH --error=/scratch/general/vast/u1266434/robustness/t5_robustness/final_models/boolq/t5-11b/seed-1/slurm_output.txt
#SBATCH --time=3-00:00:00
#SBATCH --qos=soc-gpulong-np

WORK_DIR=$HOME/scr/robustness/t5_robustness/
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
source /scratch/general/vast/u1266434/robustness/t5_robustness/envs/py385_ds_2/bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"

deepspeed t5_run_glue_trainer_boolq.py --deepspeed ds_config_zero3.json --model_name_or_path t5-11b --per_device_train_batch_size 4 --num_train_epochs 3 --seed 1 --task_name boolq --do_eval  --do_train --per_device_eval_batch_size 4 --overwrite_output_dir --output_dir final_models/boolq/t5-11b/seed-1/ --save_steps 1000000 --learning_rate 1e-4 --bf16 --predict_with_generate --max_source_length 512 | tee final_models/boolq/t5-11b/seed-1/train_output_log.txt
