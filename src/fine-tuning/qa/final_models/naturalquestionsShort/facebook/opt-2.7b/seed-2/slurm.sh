#!/bin/bash

#SBATCH --mail-user=ashimgupta95@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:2

#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np
#SBATCH --job-name=facebook_opt-2.7b_naturalquestionsShort_2
#SBATCH --output=final_models//naturalquestionsShort/facebook/opt-2.7b/seed-2/slurm_output.txt
#SBATCH --error=final_models//naturalquestionsShort/facebook/opt-2.7b/seed-2/slurm_output.txt
#SBATCH --time=12:00:00




WORK_DIR=$HOME/scr/robustness/t5_robustness/qa
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $WORK_DIR
cd $WORK_DIR
#Activate Environment
source ../envs/py385_qa//bin/activate
nvidia-smi
python -c "import torch; torch.cuda.is_available()"



echo "python finetune_gpt_qa_trainer.py --model_name_or_path facebook/opt-2.7b --dataset_name mrqa^naturalquestionsShort  --do_train --learning_rate 5e-06 --max_seq_length 512 --output_dir final_models//naturalquestionsShort/facebook/opt-2.7b/seed-2 --per_device_train_batch_size 4 --num_train_epochs 3 --doc_stride 128 --save_steps 1000000 --overwrite_output_dir --per_device_eval_batch_size 4 --num_beams 5"

python finetune_gpt_qa_trainer.py --model_name_or_path facebook/opt-2.7b --dataset_name mrqa^naturalquestionsShort  --do_train --learning_rate 5e-06 --max_seq_length 512 --output_dir final_models//naturalquestionsShort/facebook/opt-2.7b/seed-2 --per_device_train_batch_size 4 --num_train_epochs 3 --doc_stride 128 --save_steps 1000000 --overwrite_output_dir --per_device_eval_batch_size 4 --num_beams 5



