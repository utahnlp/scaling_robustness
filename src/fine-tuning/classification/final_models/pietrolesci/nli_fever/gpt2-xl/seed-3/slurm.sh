#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:8

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=gpt2-xl_pietrolesci/nli_fever_3
#SBATCH --output=final_models//pietrolesci/nli_fever/gpt2-xl/seed-3/slurm_output.txt
#SBATCH --error=final_models//pietrolesci/nli_fever/gpt2-xl/seed-3/slurm_output.txt
#SBATCH --time=3-00:00:00
#SBATCH --qos=soc-gpulong-np
#SBATCH --requeue




WORK_DIR=$HOME/scr/robustness/t5_robustness
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
source env_t5_textattack//bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"



echo "python run_glue.py --model_name_or_path gpt2-xl --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_train_epochs 3 --seed 3 --dataset_name pietrolesci/nli_fever --do_train --save_steps 1000000 --max_seq_length 512 --overwrite_output_dir --output_dir final_models//pietrolesci/nli_fever/gpt2-xl/seed-3 --learning_rate 1e-05"



python run_glue.py --model_name_or_path gpt2-xl --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_train_epochs 3 --seed 3 --dataset_name pietrolesci/nli_fever --do_train --save_steps 1000000 --max_seq_length 512 --overwrite_output_dir --output_dir final_models//pietrolesci/nli_fever/gpt2-xl/seed-3 --learning_rate 1e-05



